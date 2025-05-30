# coding=utf-8
# Copyright 2024 The Numina Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datasets
import dataclasses
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, NewType, Optional, Tuple, Union

from transformers import AutoTokenizer,  PreTrainedTokenizer


# Updated chat template for legal Q&A
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ '问题：' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '回答：' + message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '回答：' }}{% endif %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation"],
):  
    if task in ["sft", "generation"]:
        messages = example["messages"]
        
        # Apply the chat template
        example["text"] = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True if task == "generation" else False
        )
        
        # Debug: print first few examples to check formatting
        if hasattr(apply_chat_template, 'debug_count'):
            apply_chat_template.debug_count += 1
        else:
            apply_chat_template.debug_count = 1
            
        if apply_chat_template.debug_count <= 3:
            print(f"Debug example {apply_chat_template.debug_count}:")
            print(f"Messages: {messages}")
            print(f"Generated text: {example['text']}")
            print("-" * 50)
            
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation']}"
        )
    return example


def get_tokenizer(model_name_or_path, set_pad_token: bool = True) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision="main",  # Fixed typo: was "mains"
        trust_remote_code=False,
    )

    if set_pad_token is True and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    # Set the updated chat template
    tokenizer.chat_template = CHAT_TEMPLATE

    return tokenizer


def load_datasets(dataset_name_or_path):
    raw_datasets = datasets.load_from_disk(dataset_name_or_path)
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.shard(num_shards=10, index=0)
    eval_dataset = raw_datasets["test"]
    return train_dataset, eval_dataset

