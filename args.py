import os
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def parse_args():  # Parse command line arguments
    parser = ArgumentParser(description="mlm_seq")
    parser.add_argument(
        "--use_data", default=5, type=int, help="The number of datum used"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", default=500, type=int, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--lr", default=3e-4, type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--special_token_lr",
        default=1e-2,
        type=float,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--use_special_token",
        type=str2bool,
        nargs="?",
        default=True,
        help="true denotes using special token",
    )
    parser.add_argument(
        "--use_diff_lr",
        type=str,
        default="all_embed",
        choices=["special", "all_embed"],
        help="all_embed denotes using larger learning rate for all embedding layers",
    )
    parser.add_argument(
        "--use_better_init",
        type=str2bool,
        nargs="?",
        default=True,
        help="true denotes using better initialization",
    )
    parser.add_argument(
        "--noise_std_dev",
        default=1e-3,
        type=float,
        help="the variance for the random gaussian noise",
    )
    parser.add_argument(
        "--dataset",
        default="toy",
        type=str,
        choices=["internal", "gpt4", "wikidata", "wikidata-hallu", "toy"],
        help="Dataset name",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="Weight Decay for the optimizer",
    )
    parser.add_argument("--use_lora", type=str2bool, nargs="?", default=True)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument(
        "--encoder_lr", default=1e-3, type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--decoder_lr", default=1e-3, type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num_fine_tune",
        default=100,
        type=int,
        help="Number of epochs to fine tune the pre-trained BERT",
    )
    parser.add_argument(
        "--ignore_zero",
        type=str2bool,
        nargs="?",
        default=True,
        help="True denotes ignoring the padding zeros when calculating the CE loss",
    )
    parser.add_argument(
        "--norm_loss",
        type=str2bool,
        nargs="?",
        default=True,
        help="Normalize the CE loss by the number of tokens in each property",
    )
    parser.add_argument(
        "--neg_sample",
        type=str2bool,
        nargs="?",
        default=True,
        help="Use negative sampling during predicitng property key existence",
    )
    parser.add_argument(
        "--use_pretrained",
        type=str2bool,
        nargs="?",
        default=True,
        help="True denotes using pretrained model from hugging face",
    )
    parser.add_argument(
        "--only_eval",
        type=str2bool,
        nargs="?",
        default=False,
        help="Only conduct evaluation",
    )
    parser.add_argument(
        "--log_wandb",
        type=str2bool,
        nargs="?",
        default=False,
        help="True denotes using wandb to log the training process",
    )
    parser.add_argument(
        "--pretrained_model_name",
        default="t5-small",
        type=str,
        # choices=["t5-small", "t5-base", "t5-large"],
        help="Which base model to use",
    )
    parser.add_argument(
        "--decoder_choice",
        default="T5",
        type=str,
        choices=["BERT", "GPT2", "T5"],
        help="Which decoder structure we will use as the decoder",
    )
    parser.add_argument(
        "--num_optimizer",
        default="1",
        type=str,
        choices=["1", "2", "3"],
        help="Whether use separate optimizer for encoder and decoder",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seeds helps ",
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train", type=str)
    parser.add_argument("--loss_mode", choices=["sum", "mean"], default="sum", type=str)
    parser.add_argument(
        "--check_epoch", default=6, type=int, help="check_epoch for evaluation"
    )
    parser.add_argument("--top_entity", default=20, type=int, help="top_entity")
    parser.add_argument("--top_property", default=50, type=int, help="top_property")
    parser.add_argument(
        "--perturbation_test",
        default=True,
        type=str2bool,
        nargs="?",
        help="True denotes using a perturbed subset to test the model",
    )
    parser.add_argument(
        "--perturbation_exp",
        default=True,
        type=str2bool,
        nargs="?",
        help="True denotes using the perturbation testing set, "
        "and False denotes using the regular testing set",
    )
    parser.add_argument(
        "--training_mode",
        default=True,
        type=str2bool,
        nargs="?",
        help="False denotes loading saved model",
    )
    parser.add_argument(
        "--decode_type", default="at", type=str, help="AT vs NAT for decoder"
    )
    parser.add_argument(
        "--model_choice",
        default="musee",
        choices=[
            "Single-mask-Multi-Entity-Step1",
            "Single-mask-Multi-Entity-Step2",
            "Single-mask-Multi-Entity-M3",
            "generative-llm",
            "M1",
            "E1",
            "musee",
        ],
        type=str,
    )
    parser.add_argument(
        "--generative_model",
        type=str,
        default="t5-small",
        choices=["gpt2", "gpt2-large", "t5-small", "t5-base", "t5-large", "llama_3B"],
        help="Which llm model to use for the generative llm modeling",
    )
    parser.add_argument(
        "--start_sentence",
        default="\n\nCreate a JSON file containing all named entities in the previous text:\n",
        type=str,
    )
    parser.add_argument("--total_batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=0.01, type=float)
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--num_warmup_steps", default=90, type=int)
    parser.add_argument("--no_cuda", type=str2bool, nargs="?", default=False)
    parser.add_argument("--generate_num_return_sequences", default=1, type=int)
    parser.add_argument("--generate_temperature", default=0.7, type=float)
    parser.add_argument("--generate_top_k", default=50, type=int)
    parser.add_argument("--generate_top_p", default=0.95, type=float)
    parser.add_argument("--generate_do_sample", type=str2bool, nargs="?", default=False)
    parser.add_argument("--generate_num_beams", default=1, type=int)
    parser.add_argument("--evaluation_strategy", default="epoch", type=str)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--save_final_model", default=True, type=str2bool, nargs="?")
    parser.add_argument("--save_strategy", default="epoch", type=str)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="logs/logs")
    parser.add_argument(
        "--load_best_model_at_end", type=str2bool, nargs="?", default=False
    )
    parser.add_argument(
        "--torch_dtype", default="float16", type=str, choices=["float16", "float32"]
    )
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--lora_modules_to_save", default=None)
    parser.add_argument("--tight_padding", type=str2bool, nargs="?", default=True)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--saved_model_path", type=str)
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size for LLM evaluation and output generation",
    )
    parser.add_argument("--generate_output_path", type=str, default="generation_output")
    parser.add_argument("--st_checkpoint_dir", type=str, default="st_checkpoint")

    args = parser.parse_args()
    return postprocess_args(args)


def postprocess_args(args):
    curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        args.output_dir
        + f"_model_{args.model_choice}"
        + f"_{args.dataset}"
        + f"_lr_{str(args.lr)}"
        + f"_wd_{str(args.weight_decay)}"
        + f"_alpha_{str(args.alpha)}"
        + f"_lora_{str(args.use_lora)}"
        + f"_loss_mode_{str(args.loss_mode)}"
        + f"_init_{str(args.use_better_init)}"
        + f"_{str(args.mode)}",
        f"run_{curr_date}",
    )
    args.out_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return args
