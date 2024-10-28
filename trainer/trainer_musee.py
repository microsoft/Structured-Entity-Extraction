import os
import time

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn.functional import softmax
from transformers import T5Config, T5ForConditionalGeneration

from . import Trainer


def decode_logits(logits, tokenizer):
    probs = softmax(logits, dim=-1)

    # Get the most likely token IDs
    predicted_ids = torch.argmax(probs, dim=-1)
    print("predicted_ids:", predicted_ids)

    # Decode the token IDs to tokens
    decoded_tokens = []
    for i in range(predicted_ids.shape[0]):
        decoded_sequence = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
        decoded_tokens.append(decoded_sequence.split())

    return decoded_tokens


def replace_with_closest_embedding(
    predict_pk_ids, added_ent_type_tokens, added_pk_tokens, model, device
):
    # Getting the embeddings from the model and moving them to the specified device
    embeddings = model.t5_model.shared.weight.to(device)

    def find_closest_token(target_token, embeddings, allowed_tokens, device):
        target_embedding = (
            embeddings[target_token].unsqueeze(0).to(device)
        )  # Get the embedding of the target token
        allowed_embeddings = embeddings[allowed_tokens].to(
            device
        )  # Get embeddings of allowed tokens
        distances = torch.norm(target_embedding - allowed_embeddings, dim=1)
        closest_idx = distances.argmin()
        closest_token = allowed_tokens[closest_idx]

        return closest_token

    for idx, sequence in enumerate(predict_pk_ids):
        for token_idx, token in enumerate(sequence):
            if token_idx == 0:
                allowed_tokens = added_ent_type_tokens
            else:
                allowed_tokens = added_pk_tokens + [1]  # Adding end token

            closest_token = find_closest_token(
                token, embeddings, allowed_tokens, device
            )
            predict_pk_ids[idx, token_idx] = closest_token

    return predict_pk_ids


class Predictor_E_Pk_Pv(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        max_seq_length,
        max_num_entity,
        tokenizer,
    ):
        super(Predictor_E_Pk_Pv, self).__init__()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_entity = max_num_entity

        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name
        )

    def _prompt_decoder_return_logits(
        self,
        prompt_tokenized,
        input_ids,
        encoder_outputs,
        attention_mask,
        labels,
        added_ent_type_tokens=None,
        added_pk_tokens=None,
    ):
        start_token_id = self.tokenizer.pad_token_id
        end_token_id = self.tokenizer.eos_token_id
        start_tokens = torch.full(
            (labels.size(0), 1), start_token_id, dtype=torch.long, device=labels.device
        )

        combined_decoder_input_ids = torch.cat(
            [start_tokens, prompt_tokenized, labels], dim=-1
        )
        # Move all non-zero of each row to the beginning, except the first position be 0
        temp_result = torch.zeros_like(combined_decoder_input_ids)
        for i, row in enumerate(combined_decoder_input_ids):
            non_zeros = row[row != 0]  # Extract non-zero elements
            temp_result[i, :1] = 0  # Keep the first element as 0
            temp_result[i, 1 : 1 + len(non_zeros)] = (
                non_zeros  # Place non-zero elements
            )
        combined_decoder_input_ids = temp_result

        attention_mask_decoder = (
            combined_decoder_input_ids != self.tokenizer.pad_token_id
        ).long()
        attention_mask_decoder[:, 0] = 1

        # print("attention_mask_decoder:", attention_mask_decoder.sum(1))
        # print("combined_decoder_input_ids:", combined_decoder_input_ids)
        # print("attention_mask_decoder:", attention_mask_decoder)

        # if encoder_outputs is not None:
        logits = self.t5_model(
            # input_ids=input_ids,
            encoder_outputs=(
                encoder_outputs,
            ),  # convert the encoder_outputs back to a tuple
            attention_mask=attention_mask,
            decoder_input_ids=combined_decoder_input_ids,
            decoder_attention_mask=attention_mask_decoder,
        ).logits[
            :, :-1
        ]  # (batch_size, prompt_len + tgt_seq_len, vocab)
        # print("logits encoder_outputs:", logits.sum())

        # else:
        # logits = self.t5_model(
        #     input_ids=input_ids,
        #     # encoder_outputs=(encoder_outputs,),  # convert the encoder_outputs back to a tuple
        #     attention_mask=attention_mask,
        #     decoder_input_ids=combined_decoder_input_ids,
        #     decoder_attention_mask=attention_mask_decoder,
        # ).logits[:, :-1]  # (batch_size, prompt_len + tgt_seq_len, vocab)
        # print("logits input_id:", logits.sum())

        """If output is: 123000 || abc000 """
        # logits = logits[:, -labels.size(1):]  # (batch_size, tgt_seq_len, vocab)

        """If output is: 123abc || 000000 """
        # print("length:", labels.size(1))
        # Compute the start index for each row based on the number of non-zero values
        start_indices = (prompt_tokenized != 0).long().sum(dim=1)
        # print("start_indices:", start_indices)
        indices_range = torch.arange(labels.size(1)).unsqueeze(0).to(
            labels.device
        ) + start_indices.unsqueeze(1)
        # print("indices_range:", indices_range)
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).to(labels.device)
        # print("batch_indices:", batch_indices)

        logits = logits[
            batch_indices, indices_range
        ]  # (batch_size, tgt_seq_len, vocab)

        if added_ent_type_tokens is not None and added_pk_tokens is not None:
            # Apply constraints: candidates can only be {ent_type, pk, eos}
            large_negative = -1e9
            mask_token = torch.full_like(logits, large_negative).to(labels.device)
            # Apply constraints to all positions
            for token_set in [added_ent_type_tokens, added_pk_tokens, [end_token_id]]:
                mask_token[:, :, token_set] = 0

            logits += mask_token

        return logits

    def extract_entity_names(self, label_string, tokenizer):
        sep_token = "<extra_id_1>"
        sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
        token_ids = tokenizer.encode(label_string, add_special_tokens=False)

        entities = []
        current_entity = []
        for token_id in token_ids:
            if token_id == sep_token_id:
                if current_entity:
                    entity_name = tokenizer.decode(
                        current_entity, skip_special_tokens=True
                    )
                    entities.append(entity_name)
                    current_entity = []
            else:
                current_entity.append(token_id)

        return entities

    def inference_generate_ids(
        self,
        prompt_tokenized,
        input_ids,
        attention_mask,
        max_length,
        added_ent_type_tokens=None,
        added_pk_tokens=None,
    ):
        batch_size = input_ids.size(0)
        # Initialize a tensor of ones
        all_predict_ids = torch.ones(
            batch_size, max_length, dtype=torch.long, device=input_ids.device
        )

        # Determine the suppress_tokens based on added_ent_type_tokens and added_pk_tokens
        suppress_tokens = None
        if added_ent_type_tokens is not None and added_pk_tokens is not None:
            allowed_tokens = set(
                added_ent_type_tokens + added_pk_tokens + [1]
            )  # Including end token
            # print("allowed_tokens:", allowed_tokens)
            all_tokens = set(range(self.t5_model.config.vocab_size))
            suppress_tokens = list(all_tokens - allowed_tokens)

        for idx in range(batch_size):
            single_prompt = prompt_tokenized[idx, :].unsqueeze(0)
            single_cleaned_prompt = single_prompt[single_prompt != 0].unsqueeze(0)

            single_input_ids = input_ids[idx, :].unsqueeze(0)
            single_attention_mask = attention_mask[idx, :].unsqueeze(0)

            forced_decoder_ids = [
                [index + 1, element.item()]
                for index, element in enumerate(single_prompt[0])
                if element.item() != 0
            ]

            # Generate predictions with suppress_tokens if applicable
            generate_args = {
                "input_ids": single_input_ids,
                "attention_mask": single_attention_mask,
                "forced_decoder_ids": forced_decoder_ids,
                "max_length": max_length,
            }
            if suppress_tokens is not None:
                generate_args["suppress_tokens"] = suppress_tokens

            predict_ids = self.t5_model.generate(**generate_args)
            # if added_ent_type_tokens is not None and added_pk_tokens is not None:
            #     print("single_prompt:", single_prompt)
            #     print("predict_ids:", predict_ids)

            prompt_size = len(forced_decoder_ids)
            trimmed_predict_ids = predict_ids[
                :, prompt_size + 1 :
            ]  # +1 due to the first generated token always being 0

            output_length = trimmed_predict_ids.size(1)
            all_predict_ids[idx, :output_length] = trimmed_predict_ids.squeeze(0)

        return all_predict_ids

    def forward(
        self,
        input_ids,  # (b, seq_len)
        labels_ent_name,  # (b, max_num_Entity * 6)
        real_labels_ent_name,  # (b, max_num_Entity * 6)
        labels_pk,  # (b, max_num_Entity, num_all_pks+2)
        labels_pv,  # (b, max_num_Entity, num_all_pks, max_prop_len)
        attention_mask,
        attention_mask_ent_name,
        attention_mask_pk,
        attention_mask_pv,
        max_len_pv,
        device,
        added_ent_type_tokens,
        added_pk_tokens,
        mode="train",
    ):
        # Encode the input sequence once
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=False,
        )[
            0
        ]  # it is a tuple (xxx,), we use [0] to choose xxx

        """Step1: logits_ent"""
        prompt_step1 = f"predict entities"
        prompt_tokenized = (
            self.tokenizer.encode(
                prompt_step1, add_special_tokens=False, return_tensors="pt"
            )
            .repeat(input_ids.size(0), 1)
            .to(input_ids.device)
        )  # (b, x)

        if mode == "train":
            logits_ent = self._prompt_decoder_return_logits(
                prompt_tokenized,
                input_ids,
                encoder_outputs,
                attention_mask,
                labels_ent_name,
            )  # (b, max_num_Entity * 6, vocab)
        elif mode == "test":
            predict_ent_ids = self.inference_generate_ids(
                prompt_tokenized, input_ids, attention_mask, self.max_num_entity * 6
            )

        """Step2: logits_pk"""
        max_prompt_length = 20
        batch_size, seq_len = input_ids.shape

        input_ids_ent_batch = []
        enc_outputs_ent_batch = []
        attention_mask_ent_batch = []

        # Initialize a list for storing padded decoder inputs
        prompt_padded_batch_step2 = []

        for sample_idx in range(batch_size):
            if mode == "train":
                # Extract the entity names from the ground truth labels
                entity_names = self.extract_entity_names(
                    real_labels_ent_name[sample_idx], self.tokenizer
                )
            elif mode == "test":
                # Decode the predicted entity names from Step 1
                predicted_ent = [
                    self.tokenizer.decode(ids, skip_special_tokens=False)
                    for ids in predict_ent_ids
                ]
                entity_names = self.extract_entity_names(
                    predicted_ent[sample_idx], self.tokenizer
                )
                if entity_names == []:
                    entity_names = ["Fail to predict"]

            for entity_name in entity_names:
                # Format the input for the T5 decoder
                prompt_sample = f"predict type and properties {entity_name}"
                # print("prompt_sample step2:", prompt_sample)
                prompt_sample_tokenized = self.tokenizer.encode(
                    prompt_sample, add_special_tokens=False, return_tensors="pt"
                ).to(input_ids.device)

                # Pad the tokenized input to the max_decoder_length
                prompt_padded_sample = torch.nn.functional.pad(
                    prompt_sample_tokenized,
                    # (max_prompt_length - prompt_sample_tokenized.shape[1], 0),  # Add padding at the beginning
                    (0, max_prompt_length - prompt_sample_tokenized.shape[1]),
                    value=self.tokenizer.pad_token_id,
                )

                # Add to the list
                prompt_padded_batch_step2.append(prompt_padded_sample)

                # Repeat input_ids and attention_mask for each entity
                repeated_input_ids = input_ids[sample_idx].unsqueeze(0)
                repeated_enc_outputs = encoder_outputs[sample_idx].unsqueeze(0)
                repeated_attention_mask = attention_mask[sample_idx].unsqueeze(0)
                input_ids_ent_batch.append(repeated_input_ids)
                enc_outputs_ent_batch.append(repeated_enc_outputs)
                attention_mask_ent_batch.append(repeated_attention_mask)

        # Concatenate the repeated input_ids and attention masks to form a batch
        input_ids_ent_batch = torch.cat(
            input_ids_ent_batch, dim=0
        )  # (num_ent_in_batch, 512)
        enc_outputs_ent_batch = torch.cat(
            enc_outputs_ent_batch, dim=0
        )  # (num_ent_in_batch, 512, 512)
        attention_mask_ent_batch = torch.cat(
            attention_mask_ent_batch, dim=0
        )  # (num_ent_in_batch, 512)
        prompt_padded_batch_step2 = torch.cat(
            prompt_padded_batch_step2, dim=0
        )  # (num_ent_in_batch, max_prompt_length)

        if mode == "train":
            # remove all 0 tensors, since those are only for padding
            labels_pk_flatten = labels_pk[
                (labels_pk != 0).any(dim=2)
            ]  # (num_ent_in_batch, tgt_seq_len)
            logits_pk = self._prompt_decoder_return_logits(
                prompt_padded_batch_step2,
                input_ids_ent_batch,
                enc_outputs_ent_batch,
                attention_mask_ent_batch,
                labels_pk_flatten,
                added_ent_type_tokens=added_ent_type_tokens,
                added_pk_tokens=list(set(added_pk_tokens) - set(["pk_entity_name"])),
            )  # (num_ent_in_batch, tgt_seq_len, vocab)
        elif mode == "test":
            predict_pk_ids = self.inference_generate_ids(
                prompt_padded_batch_step2,
                input_ids_ent_batch,
                attention_mask_ent_batch,
                max_prompt_length + 10,
                added_ent_type_tokens=added_ent_type_tokens,
                added_pk_tokens=list(set(added_pk_tokens) - set(["pk_entity_name"])),
            )
            # print("predict_pk_ids:", predict_pk_ids)

        """Step3: logits_pv"""
        max_prompt_length = 30  # Set a suitable max length for the decoder prompts

        input_ids_pv_batch = []
        enc_outputs_pv_batch = []
        attention_mask_pv_batch = []
        prompt_padded_batch_step3 = []

        for sample_idx in range(batch_size):
            if mode == "train":
                # Extract the entity names from the ground truth labels
                entity_names = self.extract_entity_names(
                    real_labels_ent_name[sample_idx], self.tokenizer
                )
            elif mode == "test":
                # Decode the predicted entity names from Step 1
                predicted_ent = [
                    self.tokenizer.decode(ids, skip_special_tokens=False)
                    for ids in predict_ent_ids
                ]
                entity_names = self.extract_entity_names(
                    predicted_ent[sample_idx], self.tokenizer
                )
                labels_pk = predict_pk_ids.unsqueeze(0)

            for entity_idx, entity_name in enumerate(entity_names):
                # Get the entity type
                entity_type_id = labels_pk[sample_idx, entity_idx, 0]
                entity_type_token = self.tokenizer.decode([entity_type_id])

                # Iterate through each property key for the entity, starting from column index 1 in labels_pk
                for pk_idx in range(1, labels_pk.size(2) - 1):
                    pk_id = labels_pk[sample_idx, entity_idx, pk_idx]
                    if pk_id in [
                        self.tokenizer.pad_token_id,
                        self.tokenizer.eos_token_id,
                    ]:
                        continue

                    # Retrieve property key token
                    pk_token = self.tokenizer.decode([pk_id])

                    # Format the input for the T5 decoder
                    prompt_sample = f"Predict property value for {entity_name} {entity_type_token} {pk_token}"
                    # print("prompt_sample step3:", prompt_sample)
                    prompt_sample_tokenized = self.tokenizer.encode(
                        prompt_sample, add_special_tokens=False, return_tensors="pt"
                    ).to(input_ids.device)

                    # Pad the tokenized input to the max_prompt_length
                    prompt_padded_sample = torch.nn.functional.pad(
                        prompt_sample_tokenized,
                        # (max_prompt_length - prompt_sample_tokenized.shape[1], 0),
                        (0, max_prompt_length - prompt_sample_tokenized.shape[1]),
                        value=self.tokenizer.pad_token_id,
                    )

                    # Add to the list
                    prompt_padded_batch_step3.append(prompt_padded_sample)

                    # Repeat input_ids and attention_mask for each property key
                    repeated_input_ids = input_ids[sample_idx].unsqueeze(0)
                    repeated_enc_outputs = encoder_outputs[sample_idx].unsqueeze(0)
                    repeated_attention_mask = attention_mask[sample_idx].unsqueeze(0)
                    input_ids_pv_batch.append(repeated_input_ids)
                    enc_outputs_pv_batch.append(repeated_enc_outputs)
                    attention_mask_pv_batch.append(repeated_attention_mask)

        # Concatenate the repeated input_ids to form a batch
        if (
            input_ids_pv_batch != []
        ):  # Edge case for inference, sometimes the model predicts nothing for step2
            input_ids_pv_batch = torch.cat(
                input_ids_pv_batch, dim=0
            )  # (num_pair_batch, seq_len)
            enc_outputs_pv_batch = torch.cat(
                enc_outputs_pv_batch, dim=0
            )  # (num_pair_batch, seq_len)
            attention_mask_pv_batch = torch.cat(
                attention_mask_pv_batch, dim=0
            )  # (num_pair_batch, seq_len)
            prompt_padded_batch_step3 = torch.cat(
                prompt_padded_batch_step3, dim=0
            )  # (num_pair_batch, max_prompt_length)

        # if enc_outputs_pv_batch != []:  # Edge case for inference, sometimes the model predicts nothing for step2
        #
        #     attention_mask_pv_batch = torch.cat(attention_mask_pv_batch, dim=0)  # (num_pair_batch, seq_len)
        #     prompt_padded_batch_step3 = torch.cat(prompt_padded_batch_step3,
        #                                           dim=0)  # (num_pair_batch, max_prompt_length)

        if mode == "train":
            if input_ids_pv_batch != []:
                # remove all 0 tensors, since those are only for padding
                labels_pv_flatten = labels_pv[
                    (labels_pv != self.tokenizer.pad_token_id).any(dim=3)
                ]  # Flatten the labels_pv tensor
                # Predict the property values
                logits_pv = self._prompt_decoder_return_logits(
                    prompt_padded_batch_step3,
                    input_ids_pv_batch,
                    enc_outputs_pv_batch,
                    attention_mask_pv_batch,
                    labels_pv_flatten,
                )  # (num_pair_batch, max_prop_len, vocab)
            else:  # There is no other pk in addition to pk_ent_name. So we do not need to predict anything
                logits_pv = None
        elif mode == "test":
            if (
                input_ids_pv_batch != []
            ):  # Edge case for inference, sometimes the model predicts nothing for step2
                # Generate the property value predictions
                predict_pv_ids = self.inference_generate_ids(
                    prompt_padded_batch_step3,
                    input_ids_pv_batch,
                    attention_mask_pv_batch,
                    max_length=max_prompt_length + 20,
                )
            else:
                predict_pv_ids = torch.tensor([[0]])

        if mode == "train":
            return logits_ent, logits_pk, logits_pv
        elif mode == "test":
            return predict_ent_ids, predict_pk_ids, predict_pv_ids


class Trainer_E_Pk_Pv(Trainer):
    @staticmethod
    def calculate_loss(
        logits_ent,
        logits_pk,
        logits_pv,
        labels_ent_name,  # (b, max_num_Entity * 6)
        labels_pk,  # (b, max_num_Entity, num_all_pks+2)
        labels_pv,  # (b, max_num_Entity, num_all_pks, max_prop_len)
        attention_mask_ent_name,
        attention_mask_pk,
        attention_mask_pv,
        added_ent_type_tokens,
        loss_mode,
        device,
    ):
        def calculate_loss_for_step(
            logits,
            labels,
            attention_mask,
            criterion,
            loss_mode,
            added_ent_type_tokens=None,
            end_token_id=1,
            sep_token_id=32098,
        ):
            # Flatten logits and labels
            logits_flatten = logits.reshape(-1, logits.size(-1))
            labels_flatten = labels.view(-1)

            # Compute token-wise loss
            loss_tokenwise = criterion(logits_flatten, labels_flatten)

            # Apply attention mask
            loss_masked = loss_tokenwise.view_as(labels) * attention_mask

            # Initialize weights with ones
            weights = torch.ones_like(labels, dtype=torch.float, device=labels.device)

            # Increase weight for sep_token_id
            weights[labels == sep_token_id] = 2
            # weights[labels == end_token_id] = 2

            # # Increase weights for tokens in added_ent_type_tokens, if specified
            # if added_ent_type_tokens is not None:
            #     for token_id in added_ent_type_tokens:
            #         weights[labels == token_id] = 3  # Or another weight factor as desired

            # Apply weights to the loss
            loss_weighted = loss_masked * weights

            # Compute average loss
            valid_tokens_mask = attention_mask == 1
            if loss_mode == "mean":
                loss = (
                    loss_weighted.sum(dim=-1) / valid_tokens_mask.sum(dim=-1)
                ).mean()
            elif loss_mode == "sum":
                loss = (loss_weighted.sum(dim=-1) / valid_tokens_mask.sum(dim=-1)).sum()

            return loss

        criterion = torch.nn.CrossEntropyLoss(
            reduction="none"
        )  # Initialize without parameters

        """Step1: loss_ent"""
        # labels_ent_tokenized  # (b, max_num_Entity * 3)
        loss_ent = calculate_loss_for_step(
            logits_ent, labels_ent_name, attention_mask_ent_name, criterion, loss_mode
        )

        """Step2: loss_pk"""
        labels_pk_flatten = labels_pk[
            (labels_pk != 0).any(dim=2)
        ]  # (num_ent_batch, tgt_seq_len)
        attention_mask_pk_flatten = attention_mask_pk[
            (attention_mask_pk != 0).any(dim=2)
        ]  # (num_ent_batch, tgt_seq_len)
        loss_pk = calculate_loss_for_step(
            logits_pk,
            labels_pk_flatten,
            attention_mask_pk_flatten,
            criterion,
            loss_mode,
            added_ent_type_tokens,
        )

        """Step3: loss_pv"""
        if logits_pv is not None:
            labels_pv_flatten = labels_pv[
                (labels_pv != 0).any(dim=3)
            ]  # (num_pair_batch, pv_seq_len)
            attention_mask_pv_flatten = attention_mask_pv[
                (attention_mask_pv != 0).any(dim=3)
            ]  # (num_pair_batch, pv_seq_len)
            loss_pv = calculate_loss_for_step(
                logits_pv,
                labels_pv_flatten,
                attention_mask_pv_flatten,
                criterion,
                loss_mode,
            )
        else:
            loss_pv = torch.tensor(0).to(loss_pk.device)

        return loss_ent, loss_pk, loss_pv

    @staticmethod
    def compute_batch_loss(
        batch,
        model,
        added_ent_type_tokens,
        added_pk_tokens,
        loss_mode,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        input_ids = batch["input_ids"].to(device)  # (b, seq_len)
        labels_ent_name = batch["labels_ent_name"].to(device)  # (b, max_num_Entity * 6)
        real_labels_ent_name = batch["real_labels_ent_name"]  # (b, max_num_Entity * 6)
        labels_pk = batch["labels_pk"].to(device)  # (b, max_num_Entity, num_all_pks+2)
        labels_pv = batch["labels_pv"].to(
            device
        )  # (b, max_num_Entity, num_all_pks, max_prop_len)
        attention_mask = batch["attention_mask"].to(device)
        attention_mask_ent_name = batch["attention_mask_ent_name"].to(device)
        attention_mask_pk = batch["attention_mask_pk"].to(device)
        attention_mask_pv = batch["attention_mask_pv"].to(device)
        max_len_pv = attention_mask_pv.shape[-1]

        # print("labels_ent_name:", labels_ent_name)
        # print("labels_pk:", labels_pk)
        # print("labels_pv:", labels_pv)

        logits_ent, logits_pk, logits_pv = model(
            input_ids,  # (b, seq_len)
            labels_ent_name,  # (b, max_num_Entity * 6)
            real_labels_ent_name,  # (b, max_num_Entity * 6)
            labels_pk,  # (b, max_num_Entity, num_all_pks+2)
            labels_pv,  # (b, max_num_Entity, num_all_pks, max_prop_len)
            attention_mask,
            attention_mask_ent_name,
            attention_mask_pk,
            attention_mask_pv,
            max_len_pv,
            device,
            added_ent_type_tokens,
            added_pk_tokens,
        )
        (
            loss_ent,
            loss_pk,
            loss_pv,
        ) = Trainer_E_Pk_Pv.calculate_loss(
            logits_ent,
            logits_pk,
            logits_pv,
            labels_ent_name,  # (b, max_num_Entity * 6)
            labels_pk,  # (b, max_num_Entity, num_all_pks+2)
            labels_pv,  # (b, max_num_Entity, num_all_pks, max_prop_len)
            attention_mask_ent_name,
            attention_mask_pk,
            attention_mask_pv,
            added_ent_type_tokens,
            loss_mode,
            device,
        )
        return loss_ent, loss_pk, loss_pv

    @staticmethod
    def evaluate_full_dataloader(
        dataloader,
        model,
        added_ent_type_tokens,
        added_pk_tokens,
        loss_mode,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        model.eval()

        total_loss_ent, total_loss_pk, total_loss_pv, total_loss = 0, 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                loss_ent, loss_pk, loss_pv = Trainer_E_Pk_Pv.compute_batch_loss(
                    batch,
                    model,
                    added_ent_type_tokens,
                    added_pk_tokens,
                    loss_mode,
                    device,
                )
                total_loss_ent += loss_ent.item()
                total_loss_pk += loss_pk.item()
                total_loss_pv += loss_pv.item()
                total_loss += loss_ent.item() + loss_pk.item() + loss_pv.item()
        return (
            total_loss_ent / len(dataloader),
            total_loss_pk / len(dataloader),
            total_loss_pv / len(dataloader),
            total_loss / len(dataloader),
        )

    def train(
        self,
        save_path,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epochs,
        **kwargs,
    ):
        device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        log_wandb = kwargs.get("log_wandb", False)
        use_lora = kwargs.get("use_lora", "True")
        alpha = kwargs.get("alpha", 0.5)
        added_ent_type_tokens = kwargs.get("added_ent_type_tokens", None)
        added_pk_tokens = kwargs.get("added_pk_tokens", None)
        loss_mode = kwargs.get("loss_mode", None)
        report = kwargs.get("reporter", None)
        # # Start wandb
        # if log_wandb:
        #     run = wandb.init(project="your-project-name", entity="your-entity-name")

        # Initialize variables for early stopping
        no_improve_step, no_improve_epochs = 0, 0
        min_train_loss, min_val_loss = float("inf"), float("inf")

        # Compute and log the initial loss before training
        print("Monitor Epoch loss...")
        (
            avg_train_loss_ent,
            avg_train_loss_pk,
            avg_train_loss_pv,
            avg_train_loss,
        ) = Trainer_E_Pk_Pv.evaluate_full_dataloader(
            train_dataloader,
            model,
            added_ent_type_tokens,
            added_pk_tokens,
            loss_mode,
            device=device,
        )
        (
            avg_val_loss_ent,
            avg_val_loss_pk,
            avg_val_loss_pv,
            avg_val_loss,
        ) = Trainer_E_Pk_Pv.evaluate_full_dataloader(
            val_dataloader,
            model,
            added_ent_type_tokens,
            added_pk_tokens,
            loss_mode,
            device=device,
        )

        # report(epoch=0, validation_loss=avg_val_loss)
        print(f"Epoch: {0}")
        print(
            f"Train: loss_ent: {avg_train_loss_ent}, loss_pk: {avg_train_loss_pk}, loss_pv: {avg_train_loss_pv}. loss: {avg_train_loss}"
        )
        print(
            f"Val: loss_ent: {avg_val_loss_ent}, loss_pk: {avg_val_loss_pk}, loss_pv: {avg_val_loss_pv}. loss: {avg_val_loss}"
        )

        if log_wandb:
            wandb.log({"Epoch": 0})
            wandb.log(
                {
                    "train_loss_ent": avg_train_loss_ent,
                    "train_loss_pk": avg_train_loss_pk,
                    "train_loss_pv": avg_train_loss_pv,
                    "train_loss": avg_train_loss,
                }
            )
            wandb.log(
                {
                    "val_loss_ent": avg_val_loss_ent,
                    "val_loss_pk": avg_val_loss_pk,
                    "val_loss_pv": avg_val_loss_pv,
                    "val_loss": avg_val_loss,
                }
            )

        # num_step = 0
        for epoch in range(epochs):
            # if use_lora:
            #     print(f"t5_model.shared.original_module", model.t5_model.shared.original_module.weight.sum())
            #     print(f"t5_model.shared.modules_to_save", model.t5_model.shared.modules_to_save['default'].weight.sum())
            start_time = time.time()
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                (
                    loss_ent,
                    loss_pk,
                    loss_pv,
                ) = Trainer_E_Pk_Pv.compute_batch_loss(
                    batch, model, added_ent_type_tokens, added_pk_tokens, loss_mode
                )

                # batch_loss = loss_ent * alpha + loss_pk * (1 - alpha) / 2 + loss_pv * (1 - alpha) / 2
                batch_loss = loss_ent + loss_pk + loss_pv
                loss = batch_loss
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Compute loss
            print("Monitor loss at epoch...")
            (
                avg_train_loss_ent,
                avg_train_loss_pk,
                avg_train_loss_pv,
                avg_train_loss,
            ) = Trainer_E_Pk_Pv.evaluate_full_dataloader(
                train_dataloader,
                model,
                added_ent_type_tokens,
                added_pk_tokens,
                loss_mode,
                device=device,
            )
            (
                avg_val_loss_ent,
                avg_val_loss_pk,
                avg_val_loss_pv,
                avg_val_loss,
            ) = Trainer_E_Pk_Pv.evaluate_full_dataloader(
                val_dataloader,
                model,
                added_ent_type_tokens,
                added_pk_tokens,
                loss_mode,
                device=device,
            )

            print(f"Epoch: {epoch + 1}")
            print(
                f"Train: loss_ent: {avg_train_loss_ent}, loss_pk: {avg_train_loss_pk}, loss_pv: {avg_train_loss_pv}. loss: {avg_train_loss}"
            )
            print(
                f"Val: loss_ent: {avg_val_loss_ent}, loss_pk: {avg_val_loss_pk}, loss_pv: {avg_val_loss_pv}. loss: {avg_val_loss}"
            )
            report(epoch=epoch + 1, validation_loss=avg_val_loss)

            if log_wandb:
                wandb.log({"Epoch": epoch + 1})
                wandb.log(
                    {
                        "train_loss_ent": avg_train_loss_ent,
                        "train_loss_pk": avg_train_loss_pk,
                        "train_loss_pv": avg_train_loss_pv,
                        "train_loss": avg_train_loss,
                    }
                )
                wandb.log(
                    {
                        "val_loss_ent": avg_val_loss_ent,
                        "val_loss_pk": avg_val_loss_pk,
                        "val_loss_pv": avg_val_loss_pv,
                        "val_loss": avg_val_loss,
                    }
                )

            # Check for early stopping
            if avg_val_loss < min_val_loss:
                print(f"Save model... (epoch)")
                no_improve_epochs = 0
                min_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if use_lora:
                    model_path = f"{save_path}"
                    model.t5_model.shared.modules_to_save["default"] = (
                        model.t5_model.shared.original_module
                    )
                    model.save_pretrained(model_path)
                    # model = model.merge_and_unload()
                else:
                    model_path = f"{save_path}.pt"
                    torch.save(model.state_dict(), model_path)
            else:
                no_improve_epochs += 1

            if no_improve_epochs == 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            print("Time for epoch:", time.time() - start_time)
        # if log_wandb:
        #     run.finish()

    def evaluate(self, model, test_dataloader, tokenizer, **kwargs):
        pass

    @staticmethod
    def generate_full_json_output(
        model,
        dataloader,
        added_ent_type_tokens,
        added_pk_tokens,
        tokenizer,
        device,
        mode,
    ):

        def extract_elements(given_list):
            extracted = []
            for item in given_list:
                elements = item.split("<extra_id_1>")
                for element in elements:
                    cleaned_element = element.strip()
                    if cleaned_element and not cleaned_element.startswith("<"):
                        extracted.append(cleaned_element)
            return extracted

        model.eval()
        results = []

        text_index = 0

        with torch.no_grad():
            for batch in dataloader:
                start_time = time.time()
                input_ids = batch["input_ids"].to(device)  # (b, seq_len)
                # labels_ent = batch["labels_ent"].to(device)  # (b, max_num_Entity * 2)
                # labels_ent_tokenized = batch["labels_ent_tokenized"].to(device)  # (b, max_num_Entity * 3)
                real_labels_ent_name = batch[
                    "real_labels_ent_name"
                ]  # (b, max_num_Entity * 6)
                labels_ent_name = batch["labels_ent_name"].to(
                    device
                )  # (b, max_num_Entity * 6)
                labels_pk = batch["labels_pk"].to(
                    device
                )  # (b, max_num_Entity, num_all_pks+2)
                labels_pv = batch["labels_pv"].to(
                    device
                )  # (b, max_num_Entity, num_all_pks, max_prop_len)
                attention_mask = batch["attention_mask"].to(device)
                # attention_mask_ent = batch["attention_mask_ent"].to(device)
                # attention_mask_ent_tokenized = batch["attention_mask_ent_tokenized"].to(device)
                attention_mask_ent_name = batch["attention_mask_ent_name"].to(device)
                attention_mask_pk = batch["attention_mask_pk"].to(device)
                attention_mask_pv = batch["attention_mask_pv"].to(device)
                max_len_pv = attention_mask_pv.shape[-1]

                predict_ent_ids, predict_pk_ids, predict_pv_ids = model(
                    input_ids,  # (b, seq_len)
                    labels_ent_name,  # (b, max_num_Entity * 6)
                    real_labels_ent_name,  # (b, max_num_Entity * 6)
                    labels_pk,  # (b, max_num_Entity, num_all_pks+2)
                    labels_pv,  # (b, max_num_Entity, num_all_pks, max_prop_len)
                    attention_mask,
                    attention_mask_ent_name,
                    attention_mask_pk,
                    attention_mask_pv,
                    max_len_pv,
                    device,
                    added_ent_type_tokens,
                    added_pk_tokens,
                    mode="test",
                )

                # print("labels_pv:", labels_pv.shape)
                # print("predict_ent_ids:", predict_ent_ids.shape, predict_ent_ids)
                # print("predict_pk_ids:", predict_pk_ids.shape, predict_pk_ids)
                # print("predict_pv_ids:", predict_pv_ids.shape, predict_pv_ids)

                # predict_pk_ids = replace_with_closest_embedding(predict_pk_ids, added_ent_type_tokens, added_pk_tokens,
                #                                                 model, device)

                # print("predict_pk_ids:", predict_pk_ids)

                """Format prediction"""
                predict_ent_tokens = [
                    tokenizer.decode(ids, skip_special_tokens=False)
                    for ids in predict_ent_ids
                ]
                res_predict_ent_name = extract_elements(predict_ent_tokens)
                # predict_pk_tokens = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predict_pk_ids]
                # predict_pv_tokens = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predict_pv_ids]

                # Calculate the number of properties for each entity, subtracting 1 for the type
                num_properties_per_entity = [
                    (pk_ids != 1).sum().item() - 1 for pk_ids in predict_pk_ids
                ]

                # Decode predict_pk_ids
                predict_pk_tokens = [
                    tokenizer.decode(pk_ids, skip_special_tokens=True)
                    for pk_ids in predict_pk_ids
                ]

                # Initialize variables
                predict_pv_tokens = []
                start_index = 0

                # Process each entity's property values
                for num_props in num_properties_per_entity:
                    # Extract the property values for this entity
                    entity_pv_ids = predict_pv_ids[
                        start_index : start_index + num_props
                    ]

                    # Decode and handle value 1 as empty string
                    entity_pv_tokens = [
                        (
                            tokenizer.decode(pv_ids, skip_special_tokens=True)
                            if pv_ids[0] != 1
                            else ""
                        )
                        for pv_ids in entity_pv_ids
                    ]

                    # Append to the main list
                    predict_pv_tokens.append(entity_pv_tokens)

                    # Update the start index for the next entity
                    start_index += num_props

                """Format ground truth"""
                res_real_ent_name = extract_elements(real_labels_ent_name)

                labels_pk = labels_pk[(labels_pk != 0).any(dim=-1)]
                labels_pk_flat = labels_pk.view(
                    -1, labels_pk.size(-1)
                )  # Flattening to 2D
                res_real_pk = [
                    tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in labels_pk_flat
                ]

                res_real_pv = []
                for block in labels_pv[0]:  # Assuming the first dimension is always 1
                    # Filter out zero rows
                    filtered_block = block[(block != 0).any(dim=-1)]

                    # Check if the filtered block is not empty
                    if filtered_block.size(0) != 0:
                        # Decode each row in the filtered block
                        decoded_rows = [
                            tokenizer.decode(row, skip_special_tokens=True)
                            for row in filtered_block
                        ]
                        res_real_pv.append(decoded_rows)

                # Append the results of this batch to the 'batches' key in the results dictionary
                results.append(
                    {
                        "predict_ent": res_predict_ent_name,
                        "predict_pk": predict_pk_tokens,
                        "predict_pv": predict_pv_tokens,
                    }
                )

                # Optional: Print the current batch results
                print("Batch", text_index)
                print("truth_ent:", res_real_ent_name)
                print("truth_pk:", res_real_pk)
                print("truth_pv:", res_real_pv)
                print()

                print("predict_ent:", res_predict_ent_name)
                print("predict_pk:", predict_pk_tokens)
                print("predict_pv:", predict_pv_tokens)
                print("---------------------")
                text_index += 1

        return results
