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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):

        original_text_embeddings = self.t5_model.shared(input_ids)
        # Encode the input text
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # return_dict=True,
            # output_hidden_states=output_hidden_states
        )

        return original_text_embeddings, encoder_outputs

    def decode_at_emb(self, encoder_outputs, target_sequence_length=20):
        device = encoder_outputs.last_hidden_state.device
        num_position = encoder_outputs.last_hidden_state.size(0)

        # Initialize start tokens
        start_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.pad_token_id
        )
        start_tokens = torch.full(
            (num_position, 1),
            start_token_id,
            dtype=torch.long,
            device=device,
        )

        # Get the embeddings of the start tokens
        decoder_inputs_embeds = self.t5_model.get_input_embeddings()(start_tokens)

        # Initialize the decoder's attention mask with a single 1 for the start token
        decoder_attention_mask = torch.ones(
            (num_position, 1), dtype=torch.long, device=device
        )

        past_key_values = None
        return_embeds = None

        for _ in range(target_sequence_length):
            outputs = self.t5_model(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

            # Extract the last hidden state (embedding) of the last token
            last_embedding = outputs.decoder_hidden_states[-1][
                :, -1:, :
            ]  # (b, 1, d_model)
            return_embeds = (
                last_embedding
                if return_embeds is None
                else torch.cat([return_embeds, last_embedding], dim=1)
            )

            # if use past_key_values, only need to input the last decoder_inputs_embs
            decoder_inputs_embeds = last_embedding

            # Update the decoder's attention mask
            decoder_attention_mask = torch.cat(
                [
                    decoder_attention_mask,
                    torch.ones((num_position, 1), dtype=torch.long, device=device),
                ],
                dim=1,
            )

            # Store past key values for the next iteration
            past_key_values = outputs.past_key_values

        return return_embeds
