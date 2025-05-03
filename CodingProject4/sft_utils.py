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

# def prepare_cuad_for_finetuning(cuad_dataset):
#     """Convert CUAD dataset entries to the message format expected by apply_chat_template."""
#     processed_dataset = []
    
#     # Get the first example to inspect its structure
#     if len(cuad_dataset) > 0:
#         print("Example keys:", list(cuad_dataset[0].keys()))
    
#     for example in cuad_dataset:
#         try:
#             # Extract fields - update these field names based on actual dataset structure
#             contract_text = example.get("text", "")  # CUAD usually uses "text" for contract content
            
#             # Handle different possible structures for questions/answers
#             if "annotations" in example:
#                 # If annotations are available in this format
#                 for annotation in example["annotations"]:
#                     question = annotation.get("question", "")
#                     answer = annotation.get("answer", "")
                    
#                     # Format as messages for chat template
#                     messages = [
#                         {"role": "system", "content": "You are a legal assistant that helps with contract analysis."},
#                         {"role": "user", "content": f"Contract: {contract_text}\n\nQuestion: {question}"},
#                         {"role": "assistant", "content": answer}
#                     ]
                    
#                     processed_example = {"messages": messages}
#                     processed_dataset.append(processed_example)
#             else:
#                 # Fallback to direct field access if available
#                 question = example.get("question", "")
#                 answer = example.get("answer", "")
                
#                 messages = [
#                     {"role": "system", "content": "You are a legal assistant that helps with contract analysis."},
#                     {"role": "user", "content": f"Contract: {contract_text}\n\nQuestion: {question}"},
#                     {"role": "assistant", "content": answer}
#                 ]
                
#                 processed_example = {"messages": messages}
#                 processed_dataset.append(processed_example)
                
#         except Exception as e:
#             print(f"Error processing example: {e}")
#             continue
    
#     return datasets.Dataset.from_list(processed_dataset)
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
        revision="main",
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
    """Load and prepare datasets for training and evaluation."""
    print(f"Loading datasets from {dataset_name_or_path}")
    
    # Check if the path exists
    if os.path.exists(dataset_name_or_path):
        raw_datasets = datasets.load_from_disk(dataset_name_or_path)
        print(f"Dataset loaded with splits: {raw_datasets.keys()}")
    else:
        # Fallback to loading from HF Hub if path doesn't exist
        try:
            raw_datasets = datasets.load_dataset("theatticusproject/cuad")
            print(f"Dataset loaded from HF Hub with splits: {raw_datasets.keys()}")
        except Exception as e:
            print(f"Error loading dataset from HF Hub: {e}")
            raise
    
    # Print dataset stats
    for split in raw_datasets:
        print(f"Split '{split}' has {len(raw_datasets[split])} examples")
    
    # Print full structure of the first example for debugging
    if "train" in raw_datasets and len(raw_datasets["train"]) > 0:
        print("First example structure:")
        first_example = raw_datasets["train"][0]
        for key, value in first_example.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}... (truncated)")
            else:
                print(f"  {key}: {value}")
    
    # Process the datasets
    train_dataset = prepare_cuad_for_finetuning(raw_datasets["train"])
    train_dataset = train_dataset.shard(num_shards=10, index=0)
    eval_dataset = prepare_cuad_for_finetuning(raw_datasets["validation"] if "validation" in raw_datasets else raw_datasets["train"])
    
    # Verify the processed datasets have the required 'messages' field
    if len(train_dataset) > 0:
        print(f"Processed train dataset keys: {list(train_dataset[0].keys())}")
    
    return train_dataset, eval_dataset

def prepare_cuad_for_finetuning(cuad_dataset):
    """Convert CUAD dataset entries to the message format expected by apply_chat_template."""
    processed_dataset = []
    
    print(f"Processing {len(cuad_dataset)} examples from CUAD dataset")
    if len(cuad_dataset) > 0:
        print(f"Example keys in original dataset: {list(cuad_dataset[0].keys())}")
    
    for i, example in enumerate(cuad_dataset):
        # Print the first example in detail to understand structure
        if i == 0:
            print("First example content:")
            for key in example:
                value = example[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        
        try:
            # For CUAD dataset, extract title, content and provisions
            title = example.get("title", "")
            content = example.get("content", example.get("text", ""))
            
            # Fallback options based on common CUAD structures
            if not content and "contract_text" in example:
                content = example["contract_text"]
            
            # Create a default question if none exists
            default_question = "Analyze this contract and identify key legal provisions."
            
            # Try different structures to find questions and answers
            if "provisions" in example:
                for provision in example["provisions"]:
                    question = provision.get("provision", default_question)
                    answer = provision.get("answer_text", "No answer provided.")
                    
                    messages = [
                        {"role": "system", "content": "You are a legal assistant that helps with contract analysis."},
                        {"role": "user", "content": f"Contract: {content}\n\nAnalyze for: {question}"},
                        {"role": "assistant", "content": answer}
                    ]
                    
                    processed_dataset.append({"messages": messages})
            elif "annotations" in example:
                for annotation in example["annotations"]:
                    question = annotation.get("question", default_question)
                    answer = annotation.get("answer", "No answer provided.")
                    
                    messages = [
                        {"role": "system", "content": "You are a legal assistant that helps with contract analysis."},
                        {"role": "user", "content": f"Contract: {content}\n\nQuestion: {question}"},
                        {"role": "assistant", "content": answer}
                    ]
                    
                    processed_dataset.append({"messages": messages})
            else:
                # Basic fallback case - create simple dialog
                messages = [
                    {"role": "system", "content": "You are a legal assistant that helps with contract analysis."},
                    {"role": "user", "content": f"Contract: {content}\n\nQuestion: {default_question}"},
                    {"role": "assistant", "content": "This contract contains standard legal provisions."}
                ]
                
                processed_dataset.append({"messages": messages})
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    result = datasets.Dataset.from_list(processed_dataset)
    print(f"Created dataset with {len(result)} examples")
    return result