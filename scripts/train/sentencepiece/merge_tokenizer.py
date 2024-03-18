# Based on https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--llama_tokenizer_dir", default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument("--bangla_sp_model_file", default="/home/ubuntu/ccds/bangla-llama/models/bangla_sentencepiece.model", type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
bangla_sp_model_file = args.bangla_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir, trust_remote_code=True)
bangla_sp_model = spm.SentencePieceProcessor()
bangla_sp_model.Load(bangla_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
bangla_spm = sp_pb2_model.ModelProto()
bangla_spm.ParseFromString(bangla_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer), len(bangla_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add bangla tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in bangla_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"  # the path to save bangla-LLaMA tokenizer
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/bangla_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/bangla_llama.model")
tokenizer.save_pretrained(output_hf_dir)
print(f"bangla-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
bangla_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(bangla_llama_tokenizer.all_special_tokens)
print(bangla_llama_tokenizer.all_special_ids)
print(bangla_llama_tokenizer.special_tokens_map)
text = """ঢাকা দক্ষিণ এশিয়ার বাংলাদেশের রাজধানী শহর। বুড়িগঙ্গা নদীর পাশে অবস্থিত, এটি জাতীয় সরকার, বাণিজ্য এবং সংস্কৃতির কেন্দ্রে রয়েছে। Dhaka is the capital city of Bangladesh, in southern Asia. Set beside the Buriganga River, it’s at the center of national government, trade and culture."""
print("Test text:\n", text)
llama_tokenized = llama_tokenizer.tokenize(text)
bangla_llama_tokenized = bangla_llama_tokenizer.tokenize(text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenized}")
print(f"LLaMA tokenizer n_tokens={len(llama_tokenized)}")
print(f"Tokenized by bangla-LLaMA tokenizer:{bangla_llama_tokenized}")
print(f"bangla LLaMA tokenizer n_tokens={len(bangla_llama_tokenized)}")
