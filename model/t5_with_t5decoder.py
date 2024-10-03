import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


class T5_with_T5Decoder(nn.Module):
    def __init__(self, pretrained_t5_name, tokenizer, pre_train=True):
        super().__init__()

        if pre_train:
            print("Using Pre-trained T5 model")
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                pretrained_t5_name
            )
        else:
            print("Using Randomly Initialized T5 model")
            self.t5_model = T5ForConditionalGeneration(T5Config())
        print("Window Size:", self.t5_model.config.d_model)
        self.tokenizer = tokenizer
        self.d_model = self.t5_model.config.d_model

    def forward(self, input_ids=None, attention_mask=None):

        original_text_embeddings = self.t5_model.shared(input_ids)
        # Encode the input text
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return original_text_embeddings, encoder_outputs

    def decode_at(self, encoder_outputs, decoder_input_ids, target_sequence_length):
        all_logits = []
        past_key_values = None

        for i in range(target_sequence_length):
            outputs = self.t5_model(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1]
            next_tokens = next_token_logits.argmax(-1, keepdim=True)
            all_logits.append(next_token_logits.unsqueeze(1))

            # Update decoder_input_ids for the next iteration
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            past_key_values = (
                outputs.past_key_values
            )  # Store past key values for the next iteration

        return torch.cat(all_logits, dim=1)
