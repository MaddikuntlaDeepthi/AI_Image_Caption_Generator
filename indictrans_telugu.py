import os
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from sentencepiece import SentencePieceProcessor
from indicnlp.transliterate.unicode_transliterate import (
    UnicodeIndicTransliterator
)


MODEL_ID = "naklitechie/indictrans2-en-indic-dist-200M-ONNX-int8"


class IndicTransTelugu:

    def __init__(self):

        print("Loading IndicTrans2 Telugu model...")

        # Download / locate model
        self.model_path = snapshot_download(MODEL_ID)

        # ------------------------------------------
        # Load vocabularies
        # ------------------------------------------

        with open(
            os.path.join(
                self.model_path,
                "dict.SRC.json"
            ),
            "r",
            encoding="utf-8"
        ) as f:
            self.src_vocab = json.load(f)

        with open(
            os.path.join(
                self.model_path,
                "dict.TGT.json"
            ),
            "r",
            encoding="utf-8"
        ) as f:
            self.tgt_vocab = json.load(f)

        self.src_decoder = {
            v: k for k, v in self.src_vocab.items()
        }

        self.tgt_decoder = {
            v: k for k, v in self.tgt_vocab.items()
        }

        # ------------------------------------------
        # SentencePiece
        # ------------------------------------------

        self.src_spm = SentencePieceProcessor(
            model_file=os.path.join(
                self.model_path,
                "model.SRC"
            )
        )

        self.tgt_spm = SentencePieceProcessor(
            model_file=os.path.join(
                self.model_path,
                "model.TGT"
            )
        )

        # ------------------------------------------
        # Generation configuration
        # ------------------------------------------

        with open(
            os.path.join(
                self.model_path,
                "generation_config.json"
            ),
            "r",
            encoding="utf-8"
        ) as f:
            generation_config = json.load(f)

        self.decoder_start_id = int(
            generation_config.get(
                "decoder_start_token_id",
                2
            )
        )

        self.eos_id = int(
            generation_config.get(
                "eos_token_id",
                2
            )
        )

        # ------------------------------------------
        # ONNX Encoder
        # ------------------------------------------

        self.encoder = ort.InferenceSession(
            os.path.join(
                self.model_path,
                "encoder_model.onnx"
            ),
            providers=["CPUExecutionProvider"]
        )

        # ------------------------------------------
        # ONNX Decoder
        # ------------------------------------------

        self.decoder = ort.InferenceSession(
            os.path.join(
                self.model_path,
                "decoder_model.onnx"
            ),
            providers=["CPUExecutionProvider"]
        )

        # ------------------------------------------
        # ONNX Decoder with past
        # ------------------------------------------

        self.decoder_past = ort.InferenceSession(
            os.path.join(
                self.model_path,
                "decoder_with_past_model.onnx"
            ),
            providers=["CPUExecutionProvider"]
        )

        # Number of transformer layers
        self.num_layers = (
            len(self.decoder.get_outputs()) - 1
        ) // 4

        print(
            "IndicTrans2 Telugu model loaded successfully!"
        )

    # ==================================================
    # BUILD PAST KEY/VALUE INPUTS
    # ==================================================

    def _build_past_feed(self, past_outputs):

        feed = {}

        for i in range(self.num_layers):

            base = i * 4

            feed[
                f"past_key_values.{i}.decoder.key"
            ] = past_outputs[base]

            feed[
                f"past_key_values.{i}.decoder.value"
            ] = past_outputs[base + 1]

            feed[
                f"past_key_values.{i}.encoder.key"
            ] = past_outputs[base + 2]

            feed[
                f"past_key_values.{i}.encoder.value"
            ] = past_outputs[base + 3]

        return feed

    # ==================================================
    # TRANSLATE
    # ==================================================

    def translate(
        self,
        text,
        max_new_tokens=60
    ):

        if not text:
            return ""

        source_language = "eng_Latn"
        target_language = "tel_Telu"

        # ------------------------------------------
        # Add IndicTrans2 language tags
        # ------------------------------------------

        source_text = (
            source_language
            + " "
            + target_language
            + " "
            + text
        )

        # ------------------------------------------
        # SentencePiece tokenization
        # ------------------------------------------

        pieces = self.src_spm.EncodeAsPieces(
            source_text
        )

        tokens = [
            source_language,
            target_language
        ] + self.src_spm.EncodeAsPieces(text)

        unk_id = self.src_vocab["<unk>"]
        eos_source_id = self.src_vocab["</s>"]

        input_ids = []

        for token in tokens:

            if token in self.src_vocab:
                input_ids.append(
                    self.src_vocab[token]
                )
            else:
                input_ids.append(unk_id)

        input_ids.append(eos_source_id)

        input_ids = np.array(
            [input_ids],
            dtype=np.int64
        )

        attention_mask = np.ones_like(
            input_ids
        )

        # ------------------------------------------
        # Encoder
        # ------------------------------------------

        encoder_outputs = self.encoder.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        encoder_hidden_states = encoder_outputs[0]

        # ------------------------------------------
        # First decoder step
        # ------------------------------------------

        decoder_input_ids = np.array(
            [[self.decoder_start_id]],
            dtype=np.int64
        )

        generated_ids = [
            self.decoder_start_id
        ]

        past_outputs = None

        # ------------------------------------------
        # Generate
        # ------------------------------------------

        for step in range(max_new_tokens):

            if step == 0:

                decoder_outputs = self.decoder.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_attention_mask": attention_mask,
                        "encoder_hidden_states": encoder_hidden_states
                    }
                )

            else:

                past_feed = self._build_past_feed(
                    past_outputs
                )

                decoder_outputs = self.decoder_past.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_attention_mask": attention_mask,
                        **past_feed
                    }
                )

            logits = decoder_outputs[0]

            past_outputs = list(
                decoder_outputs[1:]
            )

            next_id = int(
                np.argmax(
                    logits[0, -1, :]
                )
            )

            generated_ids.append(
                next_id
            )

            if next_id == self.eos_id:
                break

            # Next decoder step receives only
            # the newly generated token
            decoder_input_ids = np.array(
                [[next_id]],
                dtype=np.int64
            )

        # ------------------------------------------
        # Remove special tokens
        # ------------------------------------------

        output_ids = generated_ids[1:]

        if self.eos_id in output_ids:

            output_ids = output_ids[
                :output_ids.index(self.eos_id)
            ]

        # ------------------------------------------
        # Convert target IDs to target tokens
        # ------------------------------------------

        output_tokens = []

        for token_id in output_ids:

            if token_id in self.tgt_decoder:

                output_tokens.append(
                    self.tgt_decoder[token_id]
                )

        # ------------------------------------------
        # Decode raw unified representation
        # ------------------------------------------

        raw_text = self.tgt_spm.DecodePieces(
            output_tokens
        )

        # ------------------------------------------
        # IndicTrans2 post-processing
        #
        # Unified Devanagari → Telugu
        # ------------------------------------------

        telugu_text = (
            UnicodeIndicTransliterator.transliterate(
                raw_text,
                "hi",
                "te"
            )
        )

        return telugu_text.strip()