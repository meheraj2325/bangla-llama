import argparse

import sentencepiece as spm


class SentencePieceEncoder:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode_text(self, input_text):
        sp = spm.SentencePieceProcessor(model_file=self.model_path)
        encoded_text = sp.encode(input_text, out_type=str)

        return encoded_text

    def run(self):
        parser = argparse.ArgumentParser(
            description="Encode text using a SentencePiece model."
        )
        parser.add_argument(
            "--model-path",
            default=self.model_path,
            help="Path to the trained SentencePiece model file.",
        )
        parser.add_argument(
            "--input-text",
            default="ঢাকা বাংলাদেশের রাজধানী শহর",
            help="Text to be encoded.",
        )

        args = parser.parse_args()
        self.model_path = args.model_path

        encoded_text = self.encode_text(args.input_text)
        print(encoded_text)


if __name__ == "__main__":
    sp_encoder = SentencePieceEncoder(
        model_path="/home/ubuntu/ccds/bangla-llama/models/bangla_sp.model"
    )  # Provide the path to your trained model here
    sp_encoder.run()
