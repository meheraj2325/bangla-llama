from transformers import LlamaTokenizer
import pandas as pd

llama_tokenizer_dir = "meta-llama/Llama-2-7b-hf"
output_hf_dir = "../tokenizer"

llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
bangla_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(bangla_llama_tokenizer.all_special_tokens)
print(bangla_llama_tokenizer.all_special_ids)
print(bangla_llama_tokenizer.special_tokens_map)

texts = ["ঢাকা দক্ষিণ এশিয়ার বাংলাদেশের রাজধানী শহর।", \
        "ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।",\
        "আজ তৃতীয় ও শেষ ওয়ানডেতে মুশফিক–রিশাদের জুটিতে জিতেছে বাংলাদেশ।",\
        "সোমালিয়ার জলদস্যুদের কবলে থাকা বাংলাদেশি জাহাজ এমভি আবদুল্লাহ", \
        "Dhaka is the capital city of Bangladesh",
        "LLaMA is a family of autoregressive large language models, released by Meta AI."]

df = pd.DataFrame()
df['input_text'] = texts

fields = {"llama_tokenizer": [], "llama_tokens_num":[], "bangla_llama_tokenizer":[], "bangla_llama_tokens_num":[]}

for text in texts:
  llama_tokenized = llama_tokenizer.tokenize(text)
  bangla_llama_tokenized = bangla_llama_tokenizer.tokenize(text)
  print(f"Tokenized by LLaMA tokenizer:{llama_tokenized}")
  print(f"LLaMA tokenizer n_tokens={len(llama_tokenized)}")
  print(f"Tokenized by bangla-LLaMA tokenizer:{bangla_llama_tokenized}")
  print(f"bangla LLaMA tokenizer n_tokens={len(bangla_llama_tokenized)}")

  fields["llama_tokenizer"].append(llama_tokenized)
  fields["llama_tokens_num"].append(len(llama_tokenized))
  fields["bangla_llama_tokenizer"].append(bangla_llama_tokenized)
  fields["bangla_llama_tokens_num"].append(len(bangla_llama_tokenized))

# print(fields)
print(len(bangla_llama_tokenizer))
print(len(llama_tokenizer))


# for key, value in fields.items():
#   df[key] = value

# print(df.head())