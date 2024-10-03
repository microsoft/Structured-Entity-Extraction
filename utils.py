import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from torch.nn import MultiheadAttention
from transformers import (AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorWithPadding, GPT2LMHeadModel,
                          GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer,
                          OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          OpenLlamaConfig, OpenLlamaForCausalLM,
                          T5ForConditionalGeneration, T5Tokenizer, Trainer,
                          TrainingArguments, TransfoXLLMHeadModel,
                          TransfoXLTokenizer)
from transformers.models.t5.modeling_t5 import T5Attention


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def plot_property_stats(
    property_accuracies, property_counts, fig_size=(10, 5), path_name=None
):
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=fig_size)

    # Sort property_counts by value, high to low
    sorted_property_counts = {
        k: v
        for k, v in sorted(
            property_counts.items(), key=lambda item: item[1], reverse=True
        )
    }

    # Bar plot with property counts
    ax1.bar(
        sorted_property_counts.keys(),
        sorted_property_counts.values(),
        color="b",
        alpha=0.5,
    )
    ax1.set_xlabel("Property Name")
    ax1.set_ylabel("Counts", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Rotate x-labels 90 degrees
    plt.xticks(rotation=90)

    # Create a second y-axis that shares the same x-axis, we already handled the x-label with ax1
    ax2 = ax1.twinx()

    # Line plot with property accuracies
    # Ensure the properties in sorted_property_accuracies follow the same order in sorted_property_counts
    sorted_property_accuracies = {
        k: property_accuracies[k] for k in sorted_property_counts.keys()
    }
    ax2.plot(
        sorted_property_accuracies.keys(),
        sorted_property_accuracies.values(),
        color="r",
    )
    ax2.set_ylabel("Accuracy", color="r")  # we already handled the x-label with ax1
    ax2.tick_params(axis="y", labelcolor="r")

    # Layout
    fig.tight_layout()

    # Save the figure if a path is provided
    if path_name is not None:
        plt.savefig(path_name, dpi=300, bbox_inches="tight")

    plt.show()


def get_generative_model_and_tokenizer(config):
    if config.saved_model_path:
        print("Loading pretrained model at", config.saved_model_path)

    if config.generative_model in ("gpt2", "gpt2-large"):
        model_path = config.saved_model_path or config.generative_model
        kwargs = {
            "pretrained_model_name_or_path": model_path,
            # "device_map": 'auto',
        }
        if hasattr(config, "torch_dtype"):
            if config.torch_dtype == "float16":
                kwargs["torch_dtype"] = torch.float16
            elif config.torch_dtype != "float32":
                raise ValueError(
                    f"torch_dtype: {config.torch_dtype} not recognized in config file."
                )
        tokenizer = GPT2Tokenizer.from_pretrained(config.generative_model)
        model = GPT2LMHeadModel.from_pretrained(**kwargs)
        tokenizer.pad_token = tokenizer.eos_token
    elif config.generative_model == "custom":
        config = OpenLlamaConfig(
            vocab_size=32000,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
        )
        model = OpenLlamaForCausalLM(config=config)
        tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
        tokenizer.pad_token_id = 0
    elif config.generative_model == "llama_3B":
        model_path = config.saved_model_path or "openlm-research/open_llama_3b_v2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs = {
            "pretrained_model_name_or_path": model_path,
            # "device_map": 'auto',
        }
        if hasattr(config, "torch_dtype"):
            if config.torch_dtype == "float16":
                kwargs["torch_dtype"] = torch.float16
            elif config.torch_dtype != "float32":
                raise ValueError(
                    f"torch_dtype: {config.torch_dtype} not recognized in config file."
                )
        model = LlamaForCausalLM.from_pretrained(**kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        # model.tie_weights()
    elif "flan-t5" in config.generative_model:
        model_path = "google/" + config.generative_model
        kwargs = {
            "pretrained_model_name_or_path": model_path,
            # "device_map": 'auto',
        }
        if hasattr(config, "torch_dtype"):
            if config.torch_dtype == "float16":
                kwargs["torch_dtype"] = torch.float16
            elif config.torch_dtype != "float32":
                raise ValueError(
                    f"torch_dtype: {config.torch_dtype} not recognized in config file."
                )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif config.generative_model in ["t5-small", "t5-base", "t5-large"]:
        model_path = config.saved_model_path or config.generative_model
        kwargs = {
            "pretrained_model_name_or_path": model_path,
        }
        if hasattr(config, "torch_dtype"):
            if config.torch_dtype == "float16":
                kwargs["torch_dtype"] = torch.float16
            elif config.torch_dtype != "float32":
                raise ValueError(
                    f"torch_dtype: {config.torch_dtype} not recognized in config file."
                )
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(config.generative_model)
        # print("T5 model", model)
        tokenizer = T5Tokenizer.from_pretrained(config.generative_model)
        # Add the new tokens to the tokenizer
        new_tokens = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "{", "}"]
        tokenizer.add_tokens(new_tokens)
    else:
        raise ValueError(f'model name "{config.generative_model}" not recognized')

    #  Apply LoRa training if config.use_lora is True
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type=(
                TaskType.SEQ_2_SEQ_LM
                if "flan-t5" in config.generative_model
                else TaskType.CAUSAL_LM
            ),
            target_modules=config.lora_target_modules,
            modules_to_save=config.lora_modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def compute_inverse_frequency_weights(entity_type_counts, num_entity_types):
    # Extract counts
    counts = list(entity_type_counts.values())
    # Compute inverse frequency
    inverse_freq = [1.0 / count for count in counts]
    # Normalize (optional, but it helps in cases where you'd want the weights to be relative to the highest class weight)
    total = sum(inverse_freq)
    normalized_weights = [freq / total for freq in inverse_freq]

    return torch.tensor(normalized_weights, dtype=torch.float32)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def get_attention_paths(model, path=""):
    paths = []
    for name, module in model.named_children():
        new_path = f"{path}.{name}" if path else name

        if isinstance(module, (T5Attention)):
            paths.append(f"{new_path}.q")
            # paths.append(f"{new_path}.k")
            paths.append(f"{new_path}.v")
            # paths.append(f"{new_path}.o")
        else:
            paths.extend(get_attention_paths(module, new_path))

    return paths


def get_transformerlayer_paths(model, path=""):
    paths = []
    for name, module in model.named_children():
        new_path = f"{path}.{name}" if path else name

        if isinstance(module, nn.TransformerEncoderLayer):
            paths.append(new_path)
        else:
            paths.extend(get_transformerlayer_paths(module, new_path))

    return paths


def remove_duplicates_and_postprocess(entity_lst):
    def postprocess(entity):
        for key, value in entity.items():
            if not isinstance(value, str):
                if isinstance(value, list):
                    try:
                        value = " ".join(value)
                    except:
                        value = str(value)
                else:
                    value = str(value)
            entity[key] = value
        return entity

    new_lst = []
    for e in entity_lst:
        e = postprocess(e)
        if not e in new_lst:
            new_lst.append(e)
    return new_lst
