
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import datasets
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from sklearn.metrics import accuracy_score
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

from huggingface_hub import HfApi, upload_folder, create_branch

device_map = "auto" #{"": int(os.environ.get("LOCAL_RANK") or 0)}
torch_dtype=torch.float16

base_model_name="meta-llama/Llama-2-7b-hf"
bangla_tokenizer_path="meta-llama/Llama-2-7b-hf"

output_dir="../../output"
checkpoint=10500
model_to_push = f"{output_dir}/checkpoint-{checkpoint}/pt_lora_model/pt_lora_model"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch_dtype,
    device_map=device_map,
    trust_remote_code=True,
)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(bangla_tokenizer_path, trust_remote_code=True)
# tokenizer.add_eos_token = True

print(f"base_model vocab size: {base_model.get_input_embeddings().weight.size(0)}")
print(f"tokenizer vocab size: {len(tokenizer)}")

# model_vocab_size = base_model.get_input_embeddings().weight.size(0)
# assert len(tokenizer) >= model_vocab_size, \
# (f"The vocab size of the tokenizer {len(tokenizer)} is smaller than the vocab size of the base model {model_vocab_size}\n"
# "This is not the intended use. Please check your model and tokenizer.")
# if model_vocab_size != len(tokenizer):
#     base_model.resize_token_embeddings(len(tokenizer))
#     print(f"Extended vocabulary size to {len(tokenizer)}")


model = PeftModel.from_pretrained(base_model, model_to_push)
model = model.merge_and_unload()

# hf_model_repo_dir = "meherajj/bangla-llama2-7b-orig-llama-tokenizer"

# model.push_to_hub(hf_model_repo_dir, use_temp_dir=True, private=True)
# tokenizer.push_to_hub(hf_model_repo_dir, use_temp_dir=True, private=True)

# # Initialize the HfApi class
# api = HfApi()

# # # Optionally, create a new branch for 'nf4'. Beware this will copy all files from main.
# # create_branch(repo_id=new_hub_model_path, repo_type="model", branch="nf4")

# # Upload the entire folder to the specified branch in the repository
# upload_folder(
#     folder_path=model_to_push,
#     repo_id=hf_model_repo_dir,
#     repo_type="model",  # Assuming it's a model; can be "dataset" or "space" as well
#     commit_message=f"Upload {checkpoint}",
#     # revision="nf4",  # Specify the branch you want to push to
#     token=True,
# )

formatted_prompt = "ঢাকা বাংলাদেশের রাজধানী এবং বৃহত্তম শহর। "

# Specify device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tokenize input prompt
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

# Get answer
# (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
outputs = model.generate(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.2,
                        repetition_penalty=1.1,
                        # num_beams=1,
                        top_k = 40,
                        top_p = 0.9,
                        )

# Decode output & print it
output  = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)

# answer_text = output[len(formatted_prompt)-1:]