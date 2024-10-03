import glob
import itertools
import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from . import DataManager


def collect_data(data_dir):
    data_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    return [f for data in data_folders for f in glob.glob(os.path.join(data, "*.json"))]


def check_bounds(idx, length):
    if idx < 0 or idx >= length:
        raise IndexError(
            f"Index {idx} is out of bounds for dataset with length {length}"
        )


def load_data_from_dir(data, idx):
    file_path = data[idx]
    with open(file_path, "r") as f:
        item = json.load(f)
    return item


class InternalDataManager(DataManager):
    class CustomDataset(Dataset):
        """ "
        A custom dataset class for loading and processing data for language modeling
        tasks.
        It returns the input_ids consisting of the token integer ids of the context
        + target json file, and label_ids, where the
        label_ids are the same as the input_ids, except that the context is masked
        out with -100 tokens.

        Inherits from the PyTorch Dataset class.
        """

        def __init__(
            self,
            tokenizer,
            file_path,
            start_sentence,
            mask_token=-100,
            length=None,
            max_seq_len=None,
            stored_in_dir=False,
            use_special_tokens=True,
            model_choice="M1",
            all_special_tokens=[],
        ):
            self.tokenizer = tokenizer
            self.file_path = file_path
            self.stored_in_dir = stored_in_dir
            self.use_special_tokens = use_special_tokens
            self.model_choice = model_choice
            self.all_special_tokens = all_special_tokens
            if not stored_in_dir and os.path.isfile(self.file_path):
                with open(self.file_path, "r") as f:
                    self.data = json.load(f)
            else:
                self.data = collect_data(self.file_path)

            # print("self.data", self.data)

            self.length = (
                min(length, len(self.data)) if length is not None else len(self.data)
            )
            if isinstance(self.data, dict):
                self.data = dict(itertools.islice(self.data.items(), self.length))
            else:
                self.data = self.data[: self.length]

            self.mask_token = mask_token
            self.start_sentence = start_sentence
            self.max_seq_len = max_seq_len

            assert (
                mask_token == -100
            )  # Otherwise the current implementation of the training pipeline breaks

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            check_bounds(idx, len(self))

            if self.stored_in_dir:
                item = load_data_from_dir(self.data, idx)
            else:
                item = self.data[list(self.data.keys())[idx]]

            if "context" in item.keys():
                context = item["context"]
            elif "raw_text" in item.keys():
                context = item["raw_text"]
            elif "description" in item.keys():
                context = item["description"].replace("\n", "")

                context = context.strip()
                context = context.encode("ascii", "ignore")
                context = context.decode()
                while context.find("  ") != -1:
                    context = context.replace("  ", " ")
                context = context.replace(" ,", ",")
                context = context.replace(" .", ".")
                context = context.replace(" ?", "?")
                context = context.replace(" !", "!")
                context = context.replace(" :", ":")
                context = context.replace(" ;", ";")
                context = context.replace("^", "")
                context = context.replace("<", "")
                context = context.replace(">", "")
                context = context.replace(" !", "!")
                context = context.replace("~", "")
                context = context.replace("\\", "")
                context = context.replace("\t", " ")
                if "{" not in self.all_special_tokens:
                    context = context.replace("{", "")
                if "}" not in self.all_special_tokens:
                    context = context.replace("}", "")
                context = context.strip()
            else:
                raise KeyError("Check the key of the textual description")

            if len(context) == 0:
                print("empty text encountered")
                return self.__getitem__(0)

            if self.model_choice == "M1":
                if "target" in item.keys():
                    target = json.dumps(item["target"])
                elif "entities" in item.keys():
                    target = json.dumps(item["entities"])
                else:
                    raise KeyError("Check the key of the target json")
            elif self.model_choice == "E1":
                if self.use_special_tokens:
                    target_json = item["entities"]
                    target = []
                    # Need special handling for new experiment 1
                    for entity_id in target_json:
                        for pk in target_json[entity_id]:
                            if pk == "type":
                                assert f"pk_{pk}" in self.all_special_tokens
                                assert (
                                    f"ent_type_{(target_json[entity_id][pk]).replace(' ', '_')}"
                                    in self.all_special_tokens
                                )
                                target.append(f"pk_{pk}")
                                target.append(
                                    f"ent_type_{(target_json[entity_id][pk]).replace(' ', '_')}"
                                )
                            else:
                                assert (
                                    f"pk_{pk.replace(' ', '_')}"
                                    in self.all_special_tokens
                                )
                                target.append(f"pk_{pk.replace(' ', '_')}")
                                target.append(target_json[entity_id][pk])
                    target = " ".join(target)
                else:
                    target = json.dumps(item["entities"])
                    # {}"":
                    target = target.replace("{", "")
                    target = target.replace("}", "")
                    target = target.replace('"', "")
                    target = target.replace(":", "")
                    target = target.replace(",", "")
            else:
                raise NotImplementedError
            # print("Description:", context)
            # print("Target:", target_json)

            # print("context:", context)
            # print("target:", target)

            if self.model_choice == "M1":
                inputs = self.tokenizer(
                    context + self.start_sentence,
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                )
            elif self.model_choice == "E1":
                inputs = self.tokenizer(
                    context,
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                )
            targets = self.tokenizer(
                target,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
                padding=True,
            )

            context_ids = inputs["input_ids"].squeeze()
            target_ids = targets["input_ids"].squeeze()

            if len(context_ids) < self.max_seq_len:
                context_ids = torch.cat(
                    [
                        context_ids,
                        torch.ones(self.max_seq_len - len(context_ids)).long()
                        * self.tokenizer.pad_token_id,
                    ],
                    dim=0,
                )
            if len(target_ids) < self.max_seq_len:
                target_ids = torch.cat(
                    [
                        target_ids,
                        torch.ones(self.max_seq_len - len(target_ids)).long() * (-100),
                    ],
                    dim=0,
                )
            assert len(context_ids) == len(target_ids)
            decoded_context = self.tokenizer.decode(
                context_ids, skip_special_tokens=True
            )
            # try:
            #     assert decoded_context == context or context in decoded_context
            # except AssertionError:
            #     print("dec context:", decoded_context)
            #     print("ori context:", context)
            #     exit(0)
            attention_mask = context_ids != self.tokenizer.pad_token_id
            label_attention_mask = target_ids != -100

            # print("context_ids:", context_ids)
            # print("target_ids:", target_ids)
            # print("attention_mask:", attention_mask)
            # print("label_attention_mask:", label_attention_mask)
            return {
                "input_ids": context_ids,
                "label_ids": target_ids,
                "attention_mask": attention_mask,
                "label_attention_mask": label_attention_mask,
            }

    class CustomDataCollator:
        """
        A custom data collator class for handling variable-length sequences during
        training.
        We need a custom one, to handle both the input_ids and label_ids returned by
        our custom dataset.
        """

        def __init__(self, tokenizer, max_length=None, tight_padding=True):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.tight_padding = tight_padding

        def __call__(
            self, batch: List[Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
            input_ids = [item["input_ids"] for item in batch]
            label_ids = [item["label_ids"] for item in batch]

            # Determine the max_length if not specified
            max_length = self.max_length or max(map(len, input_ids))
            max_padding_length = (
                min(self.max_length, max(map(len, input_ids)))
                if self.tight_padding
                else max_length
            )

            # Manually pad input_ids and label_ids with the pad_token_id
            padded_input_ids = []
            padded_label_ids = []
            masks = []
            for ids, labels in zip(input_ids, label_ids):
                # Truncate the sequences to the specified max_length
                if ids.shape[0] > max_length or labels.shape[0] > max_length:
                    print("stop")
                ids = ids[:max_length].contiguous()
                labels = labels[:max_length].contiguous()

                padding_length = max_padding_length - len(ids)
                padded_input_ids.append(
                    torch.cat(
                        (
                            torch.tensor(
                                [self.tokenizer.pad_token_id] * padding_length,
                                dtype=torch.int64,
                            ),
                            ids,
                        ),
                        dim=0,
                    )
                )
                padded_label_ids.append(
                    torch.cat(
                        (
                            torch.tensor([-100] * padding_length, dtype=torch.int64),
                            labels,
                        ),
                        dim=0,
                    )
                )
                masks.append(
                    torch.tensor(
                        [0] * padding_length + [1] * len(ids), dtype=torch.int64
                    )
                )
            padded_input_ids_tensor = torch.stack(padded_input_ids)
            padded_label_ids_tensor = torch.stack(padded_label_ids)
            attention_mask = torch.stack(masks)

            return {
                "input_ids": padded_input_ids_tensor,
                "labels": padded_label_ids_tensor,
                "attention_mask": attention_mask,
            }

    class EvalDataset(Dataset):
        """ "
        A custom dataset class for loading and processing data for language modeling
        tasks.
        It returns the input_ids consisting of the token integer ids of the context
        + target json file, and label_ids, where the
        label_ids are the same as the input_ids, except that the context is masked
        out with -100 tokens.

        Inherits from the PyTorch Dataset class.
        """

        def __init__(
            self,
            tokenizer,
            file_path,
            start_sentence,
            mask_token=-100,
            length=None,
            max_seq_len=None,
            stored_in_dir=False,
        ):
            self.tokenizer = tokenizer
            self.file_path = file_path
            self.stored_in_dir = stored_in_dir

            if not stored_in_dir and os.path.isfile(self.file_path):
                with open(self.file_path, "r") as f:
                    self.data = json.load(f)
            else:
                self.data = collect_data(self.file_path)

            self.length = (
                min(length, len(self.data)) if length is not None else len(self.data)
            )
            if isinstance(self.data, dict):
                self.data = dict(itertools.islice(self.data.items(), self.length))
            else:
                self.data = self.data[: self.length]

            self.mask_token = mask_token
            self.start_sentence = start_sentence
            self.max_seq_len = max_seq_len
            assert (
                mask_token == -100
            )  # Otherwise the current implementation of the training pipeline breaks

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            check_bounds(idx, len(self))

            if self.stored_in_dir:
                item = load_data_from_dir(self.data, idx)
            else:
                item = self.data[list(self.data.keys())[idx]]

            if "context" in item.keys():
                context = item["context"]
            elif "raw_text" in item.keys():
                context = item["raw_text"]
            elif "description" in item.keys():
                context = item["description"]
            else:
                raise KeyError("Check the key of the textual description")

            if len(context) == 0:
                print("empty text encountered")
                return self.__getitem__(0)

            input_content = self.tokenizer(
                context + self.start_sentence,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
                padding=True,
            )

            input_ids = input_content["input_ids"][: self.max_seq_len].contiguous()

            return {"input_ids": input_ids}

        def get_batch(self, start_idx, batch_size):
            check_bounds(start_idx, len(self))

            end_idx = min(len(self), start_idx + batch_size)

            input_texts = []
            for i in range(start_idx, end_idx):
                if self.stored_in_dir:
                    item = load_data_from_dir(self.data, i)
                else:
                    item = self.data[list(self.data.keys())[i]]
                if "context" in item.keys():
                    context = item["context"]
                elif "raw_text" in item.keys():
                    context = item["raw_text"]
                elif "description" in item.keys():
                    context = item["description"]
                else:
                    raise KeyError("Check the key of the textual description")

                if len(context) == 0:
                    print("empty text encountered")
                input_texts.append(context + self.start_sentence)

            tokenized_inputs = self.tokenizer(
                input_texts,
                padding=True,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": tokenized_inputs["input_ids"][
                    : self.max_seq_len
                ].contiguous(),
                "attention_mask": tokenized_inputs["attention_mask"],
            }

        def get_batch_text(self, start_idx, batch_size):
            check_bounds(start_idx, len(self))

            end_idx = min(len(self), start_idx + batch_size)

            texts = []
            for i in range(start_idx, end_idx):
                if self.stored_in_dir:
                    item = load_data_from_dir(self.data, i)
                else:
                    item = self.data[list(self.data.keys())[i]]
                if "context" in item.keys():
                    context = item["context"]
                elif "raw_text" in item.keys():
                    context = item["raw_text"]
                elif "description" in item.keys():
                    context = item["description"]
                else:
                    raise KeyError("Check the key of the textual description")

                if len(context) == 0:
                    print("empty text encountered")

                if "target" in item.keys():
                    target_json = item["target"]
                elif "entities" in item.keys():
                    target_json = item["entities"]
                else:
                    raise KeyError("Check the key of the target json")
                # print("target_json: ", target_json)
                # print("Type: ", type(target_json))
                texts.append(
                    {
                        "context": context,
                        "target": target_json,
                        "doc_id": item["doc_id"] if "doc_id" in item.keys() else str(i),
                    }
                )

            return texts

    def create_dataset(
        self,
        file_path: str,
        tokenizer,
        start_sentence: str,
        max_length: int,
        length_dataset=None,
        stored_in_dir=False,
        use_special_tokens=True,
        model_choice="M1",
        all_special_tokens=[],
    ):
        return self.CustomDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            start_sentence=start_sentence,
            mask_token=-100,
            length=length_dataset,
            max_seq_len=max_length,
            stored_in_dir=stored_in_dir,
            use_special_tokens=use_special_tokens,
            model_choice=model_choice,
            all_special_tokens=all_special_tokens,
        )

    def create_data_loader(
        self,
        file_path_train: str,
        file_path_val: str,
        file_path_test: str,
        max_length: int,
        batch_size: int,
        shuffle: bool,
        tokenizer,
        start_sentence,
        **kwargs,
    ):
        # kwargs should contain keys train_length, val_length and test_length
        datasets = {}
        for mode, file_path in zip(
            ["train", "val", "test"], [file_path_train, file_path_val, file_path_test]
        ):
            datasets[mode] = self.create_dataset(
                file_path=file_path,
                tokenizer=tokenizer,
                start_sentence=start_sentence,
                max_length=max_length,
                length_dataset=kwargs[mode + "_length"],
            )

        train_loader = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=shuffle
        )
        val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=shuffle
        )
        print("train_dataloader:", len(train_loader))
        print("val_dataloader:", len(val_loader))
        print("test_dataloader:", len(test_loader))

        return train_loader, val_loader, test_loader, datasets, tokenizer

    def create_data_collator(self, tokenizer, max_length, tight_padding):
        return self.CustomDataCollator(tokenizer, max_length, tight_padding)
