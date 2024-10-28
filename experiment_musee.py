import json

import torch
import wandb
from args import parse_args
from metrics import evaluate
from peft import LoraConfig, PeftModel, get_peft_model
from syne_tune import Reporter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils import get_attention_paths, print_trainable_parameters, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_final_json(results, ground_truth_path):
    # Load the ground truth JSON file
    with open(ground_truth_path, "r") as file:
        ground_truth_data = json.load(file)

    # Function to process each prediction and return formatted entities
    def process_prediction(ent_tokens, pk_tokens, pv_tokens):
        print("ent_tokens, pk_tokens, pv_tokens:", ent_tokens, pk_tokens, pv_tokens)
        entities = {}
        for i, (ent_name, pk_token, pv_token) in enumerate(
            zip(ent_tokens, pk_tokens, pv_tokens)
        ):
            pk_parts = pk_token.split()

            # Skip processing if pk_parts is empty
            if len(pk_parts) == 0:
                continue

            # Extracting entity type
            entity_type = (
                pk_parts[0].replace("ent_type_", "").replace("_", " ")
                if "ent_type_" in pk_parts[0]
                else "unknown"
            )

            entity_info = {"type": entity_type}
            entity_info["entity name"] = ent_name  # No need to predict entity name
            for j, key in enumerate(
                pk_parts[1:]
            ):  # Skip the first token, which is the type
                prop_key = key.replace("pk_", "").replace("_", " ")
                if "ent type" in prop_key:
                    continue
                entity_info[prop_key] = pv_token[j] if j < len(pv_token) else ""
            # if entity_type == "human":
            #     entity_info["given name"] = ent_name.split()[0]
            #     entity_info["family name"] = ent_name.split()[-1]

            entities[str(i)] = entity_info
        return entities

    # Create the final JSON object
    final_json = {}
    for doc_id, prediction in zip(ground_truth_data, results):
        ent_tokens = prediction.get("predict_ent", [])
        pk_tokens = prediction.get("predict_pk", [])
        pv_tokens = prediction.get("predict_pv", [])

        entities = process_prediction(ent_tokens, pk_tokens, pv_tokens)
        final_json[doc_id] = {
            "doc_id": doc_id,
            "description": ground_truth_data[doc_id]["description"],
            "entities": entities,
        }

    return final_json


def experiment(args):
    # Due to the forced_decoder_ids does not support batch, we have to set batch_size=1 for inference
    if args.mode == "test":
        args.batch_size = 1

    if args.model_choice == "MuSEE":
        from trainer.trainer_musee import Trainer_E_Pk_Pv

        trainer = Trainer_E_Pk_Pv()
        if args.log_wandb:
            wandb.login()
        from data.dataloader_musee import WikiDataStep1Manager

        manager = WikiDataStep1Manager()

        if args.dataset == "toy":
            data_abbrev = "toy"
            train_data_path = "data/toy/D3_toy.json"
            val_data_path = "data/toy/D3_toy.json"
            test_data_path = "data/toy/D3_toy.json"
            # test_data_path = "data/toy/dummy.json"
            use_data = 100
            max_length = 512
        elif args.dataset == "gpt4":
            data_abbrev = "d2"
            train_data_path = "data/D2_final/D2_train_final.json"
            val_data_path = "data/D2_final/D2_val_final.json"
            test_data_path = "data/D2_final/D2_test_final.json"
            use_data = 20000
            max_length = 512
        elif args.dataset == "wikidata":
            data_abbrev = "d3"
            train_data_path = "data/D3_final/D3_train_final.json"
            val_data_path = "data/D3_final/D3_val_final.json"
            test_data_path = "data/D3_final/D3_test_final.json"
            use_data = 20000
            max_length = 512
        elif args.dataset == "wikidata-hallu":
            data_abbrev = "d3"
            train_data_path = "data/D3_final/D3_train_final.json"
            val_data_path = "data/D3_final/D3_val_final.json"
            test_data_path = "data/D3_final/D3_test_final_hallu_5k.json"
            use_data = 20000
            max_length = 512
        elif args.dataset == "internal":
            data_abbrev = "inter"
            raise NotImplementedError

        # Get dataloader
        dataset, tokenizer, special_tokens_need_to_add, ent_type_tokens, pk_tokens = (
            manager.create_dataset(
                file_path=train_data_path,
                model_name=args.pretrained_model_name,
                use_data=use_data,
                max_length=max_length,
                batch_size=args.batch_size,
                shuffle=False,
                if_filter=True,
                top_entity=args.top_entity,
                top_property=args.top_property,
                data_name=data_abbrev,
            )
        )
        val_dataset, _, _, _, _ = manager.create_dataset(
            file_path=val_data_path,
            model_name=args.pretrained_model_name,
            use_data=use_data,
            max_length=max_length,
            batch_size=args.batch_size,
            shuffle=False,
            if_filter=True,
            top_entity=args.top_entity,
            top_property=args.top_property,
            data_name=data_abbrev,
        )
        test_dataset, _, _, _, _ = manager.create_dataset(
            file_path=test_data_path,
            model_name=args.pretrained_model_name,
            use_data=use_data,
            max_length=max_length,
            batch_size=args.batch_size,
            shuffle=False,
            if_filter=True,
            top_entity=args.top_entity,
            top_property=args.top_property,
            data_name=data_abbrev,
        )
        # Get the indices of the new tokens
        added_new_token_ids = tokenizer.convert_tokens_to_ids(
            special_tokens_need_to_add
        )
        added_ent_type_tokens = tokenizer.convert_tokens_to_ids(ent_type_tokens)
        added_pk_tokens = tokenizer.convert_tokens_to_ids(pk_tokens)
        print("added_new_token_ids:", added_new_token_ids)
        print("added_ent_type_tokens:", added_ent_type_tokens)
        print("added_pk_tokens:", added_pk_tokens)

        train_dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        max_seq_length = dataset.max_length
        num_entity_types = dataset.num_entity_types
        max_num_entity = dataset.max_num_entity
        num_property_keys = dataset.num_all_pks
        all_entity_types = dataset.all_entity_types
        entity_type_counts = dataset.entity_type_counts
        entity_type_counts["0"] = use_data * max_num_entity - sum(
            entity_type_counts.values()
        )
        entity_type_counts = {
            k: v
            for k, v in sorted(
                entity_type_counts.items(), key=lambda item: item[1], reverse=True
            )
        }
        property_key_counts = dataset.property_key_counts
        print("-----------")
        print("max_seq_length:", max_seq_length)
        print("num_entity_types:", num_entity_types)
        print("max_num_entity:", max_num_entity)
        print("num_property_keys:", num_property_keys)
        print("entity_type_counts:", len(entity_type_counts), entity_type_counts)
        print("property_key_counts:", property_key_counts)

        # type_weights = compute_inverse_frequency_weights(entity_type_counts, num_entity_types).to(device)
        # print("type_weights:", type_weights)

        # original_template = dataset.get_all_template().numpy()
        # all_zero_row = np.zeros(
        #     original_template.shape[1], dtype=original_template.dtype
        # )
        # template = np.vstack(
        #     (all_zero_row, original_template)
        # )  # add all-zero row for type 0
        # template = torch.tensor(template).to(device)
        # print("template:", template.shape)
        from trainer.trainer_musee import Predictor_E_Pk_Pv

        model = Predictor_E_Pk_Pv(
            pretrained_model_name=args.pretrained_model_name,
            max_seq_length=max_seq_length,
            max_num_entity=max_num_entity,
            tokenizer=tokenizer,
        ).to(device)
        model.t5_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
        print_trainable_parameters(model)

        mask_token, sep_token = "<extra_id_0>", "<extra_id_1>"
        mask_token_id = torch.tensor(
            tokenizer.encode(mask_token, add_special_tokens=False)[0]
        ).item()
        sep_token_id = torch.tensor(
            tokenizer.encode(sep_token, add_special_tokens=False)[0]
        ).item()

        vocab_size = model.t5_model.get_input_embeddings().weight.size(0)

        print("vocab_size:", vocab_size)
        print("mask_token_id:", mask_token_id)
        print("sep_token_id:", sep_token_id)
        print("--------------------")

        # # Set up wandb
        if args.log_wandb:
            run_name = "lr{}-wd{}-{}".format(args.lr, args.weight_decay, args.loss_mode)
            wandb.init(
                project="MuSEE-full-{}-{}-{}-lora-{}-init-{}".format(
                    data_abbrev,
                    args.pretrained_model_name,
                    args.use_lora,
                    args.loss_mode,
                    args.use_better_init,
                ),
                config=args,
                name=run_name,  # set the run name here
            )

        save_path = (
            f"saved/best_model/MuSEE/MuSEE_{data_abbrev}_m{args.pretrained_model_name}_"
            f"lr{args.lr}_wd{args.weight_decay}_{args.loss_mode}_lora_{args.use_lora}_init_{args.use_better_init}"
            f"best_model"
        )

        if args.use_better_init:
            print("Better initialize the special tokens' embeddings")
            print(
                "special_tokens_need_to_add:",
                len(special_tokens_need_to_add),
                special_tokens_need_to_add,
            )
            # Get the embeddings layer from the model
            embedding_layer = model.t5_model.get_input_embeddings()
            print(
                "old:",
                model.t5_model.get_input_embeddings().weight.shape,
                model.t5_model.get_input_embeddings().weight.sum(),
            )

            # Calculate new embeddings
            new_token_embeddings = []
            for token in special_tokens_need_to_add:
                # Tokenize the special token into subwords
                token = token.replace("ent_type", "")
                token = token.replace("pk", "")
                token = token.replace("_", " ")
                subtokens = tokenizer.tokenize(token)

                # Get the embeddings for the subtokens
                subtoken_ids = tokenizer.convert_tokens_to_ids(subtokens)
                subtoken_embeddings = embedding_layer.weight[subtoken_ids]

                # Calculate the average embedding
                average_embedding = subtoken_embeddings.mean(dim=0)

                # Add Gaussian noise to the average embedding
                noise = torch.randn(average_embedding.size()) * args.noise_std_dev
                new_embedding = average_embedding + noise.to(device)

                # Append to the list of new token embeddings
                new_token_embeddings.append(new_embedding)

            # Convert the list to a tensor
            new_token_embeddings = torch.stack(new_token_embeddings)

            # Set the embeddings for the new tokens in the model
            with torch.no_grad():
                # Get the indices of the new tokens
                new_token_ids = tokenizer.convert_tokens_to_ids(
                    special_tokens_need_to_add
                )
                # Update the embeddings for these tokens
                embedding_layer.weight[new_token_ids] = new_token_embeddings
            print(
                "new:",
                model.t5_model.get_input_embeddings().weight.shape,
                model.t5_model.get_input_embeddings().weight.sum(),
            )

        # Set embedding layer as trainable
        model.t5_model.shared.weight.requires_grad = True

        if args.mode == "train":
            if args.use_lora:
                target_modules = get_attention_paths(model)
                modules_to_save = ["shared"]

                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                    modules_to_save=modules_to_save,
                )

                model = get_peft_model(model, lora_config)
                print_trainable_parameters(model)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            # Set up the learning rate scheduler
            total_steps = len(train_dataloader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )
            report = Reporter()
            trainer.train(
                save_path,
                model,
                train_dataloader,
                val_dataloader,
                optimizer,
                scheduler,
                args.epochs,
                device=device,
                log_wandb=args.log_wandb,
                use_lora=args.use_lora,
                alpha=args.alpha,
                added_ent_type_tokens=added_ent_type_tokens,
                added_pk_tokens=added_pk_tokens,
                loss_mode=args.loss_mode,
                reporter=report,
            )

            if args.use_lora:
                print(
                    f"t5_model.shared.original_module",
                    model.t5_model.shared.original_module.weight.sum(),
                )
                print(
                    f"t5_model.shared.modules_to_save",
                    model.t5_model.shared.modules_to_save["default"].weight.sum(),
                )

        elif args.mode == "test":
            save_path = (
                f"saved/best_model/MuSEE/MuSEE_{data_abbrev}_m{args.pretrained_model_name}_"
                f"lr{args.lr}_wd{args.weight_decay}_{args.loss_mode}_lora_{args.use_lora}_init_{args.use_better_init}"
                f"best_model"
            )
            print("save_path:", save_path)
            if args.use_lora:
                model = PeftModel.from_pretrained(model, save_path)
                print(
                    f"t5_model.shared.original_module",
                    model.t5_model.shared.original_module.weight.sum(),
                )
                print(
                    f"t5_model.shared.modules_to_save",
                    model.t5_model.shared.modules_to_save["default"].weight.sum(),
                )
                model = model.merge_and_unload()
            else:
                model.load_state_dict(
                    torch.load(f"{save_path}.pt", map_location=device)
                )

            print(
                "after load pretrained (get_input_embeddings):",
                model.t5_model.get_input_embeddings().weight.shape,
                model.t5_model.get_input_embeddings().weight.sum(),
            )

            print(
                "after load pretrained (shared):",
                model.t5_model.shared.weight.shape,
                model.t5_model.shared.weight.sum(),
            )

            model.eval()
            # generate json output
            # id2entity, id2property = dataset.id2entity(), dataset.id2property()
            results = Trainer_E_Pk_Pv.generate_full_json_output(
                model,
                test_dataloader,
                added_ent_type_tokens,
                added_pk_tokens,
                tokenizer,
                device,
                mode=args.mode,
            )

            final_json = generate_final_json(results, test_data_path)
            print("final_json:", json.dumps(final_json, indent=4))

            # Save to a JSON file
            prediction_path = (
                f"saved/best_model/MuSEE/saved_json/MuSEE_{data_abbrev}_m{args.pretrained_model_name}_"
                f"lr{args.lr}_wd{args.weight_decay}_{args.loss_mode}_lora_{args.use_lora}_init_{args.use_better_init}.json"
            )
            with open(prediction_path, "w", encoding="utf-8") as file:
                json.dump(final_json, file, ensure_ascii=False, indent=4)

            metrics = evaluate(test_data_path, prediction_path)


def run():
    args = parse_args()
    set_seed(args.seed)
    experiment(args)


if __name__ == "__main__":
    run()
