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


CHAT_TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'system')%}{{ '' }}{% elif (message['role'] == 'user')%}{{ '### Problem: ' + message['content'] + '\n' }}{% elif (message['role'] == 'assistant')%}{{ '### Solution: ' + message['content'] + '\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Solution: ' }}{% endif %}{% endfor %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation']}"
        )
    return example



def get_tokenizer(model_name_or_path, set_pad_token: bool = True) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision="mains",
        trust_remote_code=False,
    )

    if set_pad_token is True and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    tokenizer.chat_template = CHAT_TEMPLATE

    return tokenizer


def load_datasets(dataset_name_or_path):
    raw_datasets = datasets.load_from_disk(dataset_name_or_path)
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.shard(num_shards=10, index=0)
    eval_dataset = raw_datasets["test"]
    return train_dataset, eval_dataset

