from dataclasses import dataclass, field
from typing import List, Optional

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelConfig:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adapters.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: Optional[str] = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can use --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    lora_use_dora: bool = field(
        default=False,
        metadata={"help": ("Flag to toggle the use of DoRA training.")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")



@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The dataset to train on.")},
    )
    dataset_mode: Optional[str] = field(
        default="online",
        metadata={"help": ("The dataset mode to use.")},
    )
    benchmarks: List[str] = field(
        default_factory=lambda: [], metadata={"help": ("The benchmarks to run after training.")}
    )
    mask_user_turns: bool = field(
        default=False,
        metadata={"help": ("Whether to mask user turns.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    neftune_noise_alpha: Optional[float] = field(
        default=None, metadata={"help": ("If not `None`, this will activate NEFTune noise embeddings.")}
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")
    overwrite_hub_revision: bool = field(default=False, metadata={"help": ("Whether to overwrite the Hub revision.")})
    packing: bool = field(
        default=True, metadata={"help": ("Whether to pack sequences of the dataset for faster training.")}
    )
    push_to_hub_revision: bool = field(default=False, metadata={"help": ("Whether to push to a Hub revision/branch.")})
    # reward_loss_fn: Optional[str] = field(
    #     default="NegLogSigmoid",
    #     metadata={"help": ("Loss function for reward model.")},
    # )
    quants: List[str] = field(
        default_factory=lambda: [], metadata={"help": ("Which quantization methods to apply on final model.")}
    )
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=0.1)
    save_total_limit: Optional[int] = field(default=1)
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Tags to group and filter runs on Weights and Biases.")},
    )
    wandb_enabled: bool = field(
        default=True,
        metadata={"help": ("Whether to enable or disable WandB.")},
    )
    wandb_project: Optional[str] = field(
        default="h4",
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_entity: Optional[str] = field(
        default="huggingface",
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default="numina-math-sft",
        metadata={"help": ("Group multiple runs under this group name.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Set this to a globally unique string (per project) corresponding to a single run of your script."
            )
        },
    )

