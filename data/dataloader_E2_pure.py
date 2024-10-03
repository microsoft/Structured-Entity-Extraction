import json
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import T5Tokenizer

from . import DataManager


class WikiDataStep1Manager(DataManager):
    class WikiDatasetStep1Filtered(Dataset):
        def __init__(
            self,
            data,
            tokenizer,
            max_length,
            top_entity=None,
            top_property=None,
            data_name=None,
        ):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            # self.max_num_entity = max(
            #     len(item.get("entities", {})) for item in data.values()
            # )
            if data_name in ["d3", "d3-hallu"]:
                self.max_num_entity = 20
            elif data_name in ["d2", "d2-hallu"]:
                self.max_num_entity = 10
            elif data_name == "toy":
                self.max_num_entity = 4
            self.max_entities = max(len(item["entities"]) for item in data.values())
            self.max_prop = max(
                len(item["entities"][entity])
                for item in data.values()
                for entity in item["entities"]
            )
            self.max_prop_length = 15

            # Initialize storage variables
            self.all_entity_types = []
            self.all_pks = []
            self.entity_type_counts = {}
            self.property_key_counts = {}

            self._compute_counts()

            # Filter by top entity and property
            if top_entity:
                self._filter_top_entities(top_entity)
            if top_property:
                self._filter_top_properties(top_property)

            self._filter_data()

            self.num_entity_types = len(self.all_entity_types)
            self.num_all_pks = len(self.all_pks)

            self._build_template()

        def _compute_counts(self):
            """Compute counts for entity types and property keys."""
            for item in self.data.values():
                for entity in item["entities"].values():
                    # Exclude the "pk_type" key
                    self.all_pks.extend([k for k in entity.keys() if k != "pk_type"])
                    entity_type = entity["pk_type"]
                    self.all_entity_types.append(entity_type)

                    self.entity_type_counts[entity_type] = (
                        self.entity_type_counts.get(entity_type, 0) + 1
                    )
                    for pk in entity.keys():
                        if pk != "pk_type":
                            self.property_key_counts[pk] = (
                                self.property_key_counts.get(pk, 0) + 1
                            )

            # Sorting entity_type_counts from high to low
            self.entity_type_counts = dict(
                sorted(
                    self.entity_type_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            # Sorting property_key_counts from high to low
            self.property_key_counts = dict(
                sorted(
                    self.property_key_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            # Sorting all_entity_types based on entity_type_counts order
            self.all_entity_types = sorted(
                self.all_entity_types,
                key=lambda x: self.entity_type_counts[x],
                reverse=True,
            )

            # Sorting all_pks based on property_key_counts order
            self.all_pks = sorted(
                self.all_pks, key=lambda x: self.property_key_counts[x], reverse=True
            )

        def _filter_top_entities(self, top_entity):
            """Filter top entities."""
            sorted_entities = sorted(
                self.entity_type_counts.items(), key=lambda x: x[1], reverse=True
            )
            self.all_entity_types = [e[0] for e in sorted_entities[:top_entity]]
            print(f"top_sorted_entities counts: {dict(sorted_entities[:top_entity])}\n")

        def _filter_top_properties(self, top_property):
            """Filter top properties."""
            sorted_properties = sorted(
                self.property_key_counts.items(), key=lambda x: x[1], reverse=True
            )
            self.all_pks = [p[0] for p in sorted_properties[:top_property]]
            print(
                f"top_sorted_properties counts: {dict(sorted_properties[:top_property])}\n"
            )

        def _filter_data(self):
            """Filter data items and entities."""
            # Filter out items with no entities of desired types
            keys_to_remove = [
                key
                for key, item in self.data.items()
                if not any(
                    e["pk_type"] in self.all_entity_types
                    for e in item["entities"].values()
                )
            ]
            for key in keys_to_remove:
                del self.data[key]

            # Filter out entities not having any of the selected top properties
            for item in self.data.values():
                entities_to_remove = [
                    entity_key
                    for entity_key, entity in item["entities"].items()
                    if not (set(entity.keys()) - {"pk_type"}) & set(self.all_pks)
                ]
                for entity_key in entities_to_remove:
                    del item["entities"][entity_key]

            # Remove items with no entities or only with "0" type
            keys_to_remove = [
                key
                for key, item in self.data.items()
                if not item.get("entities")
                or all(
                    entity["pk_type"] == 0 for entity in item.get("entities").values()
                )
            ]
            for key in keys_to_remove:
                del self.data[key]

        def _build_template(self):
            """Post filtering setup to determine unique entity types and property keys."""
            # self.all_entity_types = sorted(list(set(self.all_entity_types)))
            # self.all_pks = sorted(list(set(self.all_pks)))

            # Create a template tensor to store property presence info
            self.template = torch.zeros(
                (self.num_entity_types, self.num_all_pks), dtype=torch.int
            )
            for item in self.data.values():
                for entity_name in item["entities"]:
                    entity = item["entities"][entity_name]
                    if entity["pk_type"] in self.all_entity_types:
                        for pk in entity:
                            if pk in self.all_pks:
                                self.template[
                                    self.all_entity_types.index(entity["pk_type"])
                                ][self.all_pks.index(pk)] = 1
            print("template sum by E_type:", self.template.sum(1))
            print("template sum by Pk:", self.template.sum(0))

        def __len__(self):
            return len(self.data)

        def id2entity(self):
            # index 0 is for padded (not-real) entity
            return {i + 1: entity for i, entity in enumerate(self.all_entity_types)}

        def id2property(self):
            return {i: property_key for i, property_key in enumerate(self.all_pks)}

        def get_entity_label(self):
            entity_label = {}
            for i, e in enumerate(self.all_entity_types):
                entity_label[e] = i + 1
            entity_label["padding"] = 0
            return entity_label

        def get_all_template(self):
            return self.template

        def get_template(self, entity_type: str):
            return self.template[self.all_entity_types.index(entity_type)]

        def sort_entity_values(self, entity_values):
            # Function to get ID for a key
            def get_id(key):
                return self.tokenizer.convert_tokens_to_ids(key)

            # Excluding 'pk_type' and sorting the remaining keys based on their IDs
            sorted_keys = sorted(
                [key for key in entity_values if key != "pk_type"], key=get_id
            )

            # Constructing the sorted dictionary
            sorted_dict = {"pk_type": entity_values["pk_type"]}
            for key in sorted_keys:
                sorted_dict[key] = entity_values[key]

            return sorted_dict

        def __getitem__(self, idx):
            item = self.data[list(self.data.keys())[idx]]
            description = item.get("description", None)
            entities = item.get("entities", None)

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
            context = context.replace("&lt;", "")
            context = context.replace("&gt;", "")
            context = context.replace(" !", "!")
            context = context.replace("~", "")
            context = context.replace("\\", "")
            context = context.replace("\t", " ")
            context = context.replace("{", "")
            context = context.replace("}", "")
            context = context.strip()

            """
            ### Obtain input_ids and attention mask
            """
            src_text = f"{description}"
            src_tokenized = self.tokenizer.encode_plus(
                src_text,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            input_ids = src_tokenized["input_ids"].flatten()  # (seq_len, )
            attention_mask = src_tokenized["attention_mask"].flatten()  # (seq_len, )

            # """
            # ### Step 1: Entity positions
            # """
            # # Tokenize the description text
            # tokenized_text = self.tokenizer.tokenize(description)
            # labels_ent = torch.full((self.max_num_entity * 2,), -1)  # Using -1 for padding
            #
            # # Sort the entities by the length of the entity name in descending order
            # sorted_entities_by_len = sorted(entities.items(), key=lambda item: len(item[1]['pk_entity_name']),
            #                                 reverse=True)
            #
            # entity_spans = []
            #
            # # Search and replace process
            # for i, (entity_key, entity_values) in enumerate(sorted_entities_by_len):
            #     if i >= self.max_num_entity:
            #         break
            #
            #     entity_name = entity_values['pk_entity_name']
            #     entity_tokens = self.tokenizer.tokenize(entity_name)
            #
            #     # Find the closest match in tokenized text
            #     start_token_pos = -1
            #     min_distance = len(entity_tokens)  # Maximum possible distance
            #     for idx in range(len(tokenized_text) - len(entity_tokens) + 1):
            #         window = tokenized_text[idx:idx + len(entity_tokens)]
            #         distance = sum(1 for a, b in zip(window, entity_tokens) if a != b)
            #         if distance < min_distance:
            #             start_token_pos = idx
            #             min_distance = distance
            #
            #     # Consider it a match if min_distance is within an acceptable threshold
            #     if start_token_pos != -1:
            #         end_token_pos = start_token_pos + len(entity_tokens) - 1
            #         entity_spans.append((start_token_pos, end_token_pos, i))
            #
            #         # Replace entity tokens with placeholders
            #         placeholder_tokens = ["<extra_id_{}>".format(j) for j in range(len(entity_tokens))]
            #         tokenized_text[start_token_pos:end_token_pos + 1] = placeholder_tokens
            #     else:
            #         entity_spans.append((-1, -1, i))
            #
            # # Sort the spans based on their start position
            # filtered_spans = [span for span in entity_spans if span[0] != -1 and span[1] != -1]
            # sorted_spans = sorted(filtered_spans)
            #
            # # Reorder the spans and entities based on the sorted order
            # for sorted_index, span_info in enumerate(sorted_spans):
            #     original_index = span_info[2]
            #     labels_ent[2 * sorted_index] = span_info[0]
            #     labels_ent[2 * sorted_index + 1] = span_info[1]
            #
            # #     sorted_entities_by_len[sorted_index] = sorted_entities_by_len[original_index]
            # # Tokenize positions to create labels suitable for the T5 model
            # def position_to_token(pos):
            #     return self.tokenizer.pad_token_id if pos == -1 else str(pos.item())
            #
            # labels_ent_tokens = [position_to_token(pos) for pos in labels_ent if pos != -1]
            #
            # # Tokenize the sequence of position tokens with separation tokens
            # labels_ent_tokenized = self.tokenizer(
            #     ' '.join(labels_ent_tokens),
            #     max_length=self.max_num_entity * 3,
            #     padding='max_length',
            #     truncation=True,
            #     return_tensors='pt'
            # )['input_ids'].squeeze(0)
            #
            # sorted_entities_by_len = [sorted_entities_by_len[span_info[2]] for span_info in sorted_spans]
            # sorted_entities_by_occur = {str(index): value for index, (_, value) in enumerate(sorted_entities_by_len)}

            # Extracting entity names and formatting them with [SEP] token
            entity_names = [entity["pk_entity_name"] for entity in entities.values()]
            entity_names = entity_names[
                : self.max_num_entity
            ]  # only process max_num_entity
            real_labels_ent_name = (
                "<extra_id_1> " + " <extra_id_1> ".join(entity_names) + " <extra_id_1>"
            )
            labels_ent_name = self.tokenizer(
                real_labels_ent_name,
                max_length=self.max_num_entity * 6,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            # print("real_labels_ent_name:", real_labels_ent_name)
            # print("labels_ent_name:", labels_ent_name)

            # Create the attention mask for the entity positions
            # attention_mask_ent = (labels_ent != -1).long()
            # attention_mask_ent_tokenized = (labels_ent_tokenized != 0).long()
            attention_mask_ent_name = (labels_ent_name != 0).long()

            """
            ### Labels for Step2: Obtain labels for Entity type and property key
            """
            end_token_id = self.tokenizer.eos_token_id
            labels_pk = torch.full(
                (self.max_num_entity, self.num_all_pks + 2), self.tokenizer.pad_token_id
            )  # (max_num_Entity, num_all_pks+2)

            for i, (_, entity_values) in enumerate(entities.items()):
                entity_values = self.sort_entity_values(entity_values)
                if (
                    i >= self.max_num_entity
                ):  # We only process up to max_num_entity entities
                    break

                # Get the special token ID for the entity type
                ent_type_id = self.tokenizer.convert_tokens_to_ids(
                    entity_values["pk_type"]
                )
                labels_pk[i, 0] = ent_type_id

                # Use a counter for the position in the label_ids tensor
                pk_counter = 1

                # Iterate over properties
                for pk in entity_values:
                    if pk not in [
                        "pk_type",
                        "pk_entity_name",
                    ]:  # type is already added. No need to predict ent_name
                        # Use the property key as a special token to get its ID
                        pk_id = self.tokenizer.convert_tokens_to_ids(pk)
                        if (
                            pk_counter < labels_pk.size(1) - 1
                        ):  # Leave space for the end token
                            labels_pk[i, pk_counter] = pk_id
                            pk_counter += 1

                # Add the end token if there's at least one property key
                if pk_counter < labels_pk.size(1):
                    labels_pk[i, pk_counter] = end_token_id

            # The attention mask is binary: 1 for special tokens, 0 for padding tokens
            attention_mask_pk = (
                labels_pk != self.tokenizer.pad_token_id
            ).long()  # (max_num_Entity, num_all_pks+2)

            """
            ### Labels for Step3: Property values (max_num_Entity, num_all_pks, max_prop_len)
            """
            labels_pv = torch.full(
                (self.max_num_entity, self.num_all_pks, self.max_prop_length),
                self.tokenizer.pad_token_id,
            )  # Initialize with padding token ID

            for i, (_, entity_values) in enumerate(entities.items()):
                entity_values = self.sort_entity_values(entity_values)
                if (
                    i >= self.max_num_entity
                ):  # Only process up to max_num_entity entities
                    break

                # Counter for the property key
                pk_counter = 0
                for pk in entity_values:
                    if pk not in [
                        "pk_type",
                        "pk_entity_name",
                    ]:  # type is already added. No need to predict ent_name
                        # Encode the property value
                        encoded_prop = self.tokenizer.encode(
                            entity_values[pk]
                            + self.tokenizer.eos_token,  # Encode the property value
                            add_special_tokens=False,
                            max_length=self.max_prop_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        ).flatten()
                        # Place the encoded property value in the tensor
                        labels_pv[i, pk_counter, :] = encoded_prop
                        pk_counter += 1

            # Create a mask for the encoded properties
            attention_mask_pv = (labels_pv != self.tokenizer.pad_token_id).long()

            return {
                "input_ids": input_ids,  # (seq_len, )
                # "labels_ent": labels_ent,  # (max_num_Entity * 2, )
                # "labels_ent_tokenized": labels_ent_tokenized,  # (max_num_Entity * 3, )
                "labels_ent_name": labels_ent_name,  # (max_num_Entity * 6, )
                "real_labels_ent_name": real_labels_ent_name,  # (max_num_Entity * 6, )
                "labels_pk": labels_pk,  # (max_num_Entity, num_all_pks+2)
                "labels_pv": labels_pv,  # (max_num_Entity, num_all_pks, max_prop_len)
                "attention_mask": attention_mask,
                # "attention_mask_ent": attention_mask_ent,
                # "attention_mask_ent_tokenized": attention_mask_ent_tokenized,
                "attention_mask_ent_name": attention_mask_ent_name,
                "attention_mask_pk": attention_mask_pk,
                "attention_mask_pv": attention_mask_pv,
            }

    def create_dataset(
        self, file_path: str, use_data: int = None, max_length: int = 1024, **kwargs
    ):
        pretrained_model_name = kwargs.get("model_name", None)
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
        special_tokens_need_to_add = []
        ent_type_tokens = []  # List for tokens starting with "ent_type_"
        pk_tokens = []  # List for tokens starting with "pk_"

        with open(file_path, "r") as f:
            data = json.load(f)

            # Modify the keys in the data
            for k, entities in data.items():
                for entity_id, entity in entities["entities"].items():
                    new_entity = {}
                    for prop_key, prop_value in entity.items():
                        if prop_key == "type":
                            new_key = f"pk_{prop_key}"
                            new_value = f'ent_type_{prop_value.replace(" ", "_")}'
                            new_entity[new_key] = new_value
                            special_tokens_need_to_add.append(new_value)
                            ent_type_tokens.append(new_value)  # Add to ent_type_tokens
                        # elif prop_key != "entity name": # we do not need to train / predict entity name for pk and pv
                        else:
                            new_key = f'pk_{prop_key.replace(" ", "_")}'
                            new_entity[new_key] = prop_value
                            special_tokens_need_to_add.append(new_key)
                            pk_tokens.append(new_key)  # Add to pk_tokens
                    entities["entities"][entity_id] = new_entity

        special_tokens_need_to_add = sorted(list(set(special_tokens_need_to_add)))
        ent_type_tokens = sorted(
            list(set(ent_type_tokens))
        )  # Remove duplicates and sort
        pk_tokens = sorted(list(set(pk_tokens)))  # Remove duplicates and sort
        # print("special_tokens_need_to_add:", len(special_tokens_need_to_add), special_tokens_need_to_add)
        tokenizer.add_tokens(special_tokens_need_to_add)

        print("Full data length:", len(data))
        data = dict(list(data.items())[:use_data])
        print("Used data length:", len(data))

        top_entity = kwargs.get("top_entity", None)
        top_property = kwargs.get("top_property", None)
        data_name = kwargs.get("data_name", None)
        dataset = self.WikiDatasetStep1Filtered(
            data,
            tokenizer,
            max_length=max_length,
            top_entity=top_entity,
            top_property=top_property,
            data_name=data_name,
        )
        print("Max_num_entity:", dataset.max_num_entity)
        print("Max_num_pk:", dataset.num_all_pks)
        print("Max_prop_len:", dataset.max_prop_length)
        print("*************")

        return (
            dataset,
            tokenizer,
            special_tokens_need_to_add,
            ent_type_tokens,
            pk_tokens,
        )

    def create_data_loader(
        self,
        file_path: str,
        use_data: int,
        max_length: int,
        batch_size: int,
        shuffle: bool,
        **kwargs,
    ):
        dataset, tokenizer, _, _, _ = self.create_dataset(
            file_path, use_data, max_length, **kwargs
        )

        if len(dataset) < 10:
            train_size = len(dataset) - 2
            val_size = 1
            test_size = 1
        else:
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        print("train_dataloader:", len(train_loader))
        print("val_dataloader:", len(val_loader))
        print("test_dataloader:", len(test_loader))

        return train_loader, val_loader, test_loader, dataset, tokenizer
