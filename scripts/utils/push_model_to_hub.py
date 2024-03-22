
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


output_dir="../../../output"
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
torch_dtype=torch.float16
model_name="meta-llama/Llama-2-7b-hf"
new_model = f"{output_dir}/checkpoint-12650"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.add_eos_token = True


new_model_name = "rahat01/bangla-llama-7b-base-v0.0.2"

model.push_to_hub(new_model_name, use_temp_dir=True, private=True)
tokenizer.push_to_hub(new_model_name, use_temp_dir=True, private=True)
