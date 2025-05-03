import logging
import math
import random
import sys
import time
import os

import datasets
import torch
import transformers
import wandb
from accelerate import Accelerator
from configs import SFTConfig
from sft_utils import (
    apply_chat_template,
    get_tokenizer,
    load_datasets
)
from transformers import set_seed, HfArgumentParser
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def main():
    accelerator = Accelerator()

    parser = HfArgumentParser(SFTConfig)
    sft_config = parser.parse_yaml_file("sft_config.yaml")[0]
    model_name_or_path = sft_config.model_name_or_path
    set_seed(sft_config.seed)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_name_or_path=model_name_or_path, set_pad_token=sft_config.packing)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")


    model_kwargs = dict(
        revision=False,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False if sft_config.gradient_checkpointing else True,
    )
 

    #######################################
    # Load datasets and apply chat template
    #######################################
    
    if sft_config.dataset_mode == "local":
        train_dataset, eval_dataset = load_datasets(sft_config.dataset_name_or_path)
    else:
        train_dataset = datasets.load_dataset(sft_config.dataset_name_or_path, split="train")
        eval_dataset = datasets.load_dataset(sft_config.dataset_name_or_path, split="test")

    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})
    eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})

    time_string = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
    save_path = f"{sft_config.output_dir}/model_{time_string}"
    os.makedirs(save_path, exist_ok=True)

    assert len(train_dataset) <= 100_000
    assert len(eval_dataset) <= 100
    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=sft_config,
        train_dataset=train_dataset if sft_config.do_train else None,
        eval_dataset=eval_dataset if sft_config.do_eval else None,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=sft_config.packing,
    )

    ###############
    # Training loop
    ###############
    if sft_config.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    ##########
    # Evaluate
    ##########
    if sft_config.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (len(eval_dataset))
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(save_path)
    logger.info(f"Model saved to {save_path}")

    # Save everything else on main process
    if accelerator.is_main_process:
        trainer.model.config.save_pretrained(save_path)



if __name__ == "__main__":
    main()
