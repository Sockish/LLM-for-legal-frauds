# Coding Project: Fine-Tuning Llama-3.2-1B for Math Problem Solving with SFT

## Overview
In this project, you will fine-tune the pre-trained Llama-3.2-1B model using Supervised Fine-Tuning (SFT) to solve math problems. The model will be trained to solve problems from your own dataset, which containing no more than 100,000 examples. A key challenge in this task is how to curate and structure high-quality data to maximize the effectiveness of fine-tuning. The final goal is to test the accuracy of the model's ability to generate correct answers to math problems.

## Environment Setup
```shell
conda create -n sft python=3.10 && conda activate sft
```

Next please install PyTorch `v2.1.2`.

You can then install the remaining package dependencies as follows:

```shell
pip install -r requirements.txt
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

## Dataset
The codebase uses a subset of the **AI-MO/NuminaMath-CoT** dataset, which containing approximately 86,000 examples. However, you are allowed to collect **your own** training set, which can contain up to **100,000 examples** and the validation set will contain up to **100 examples**. Each data example must not exceed **2048 tokens**.

### Training Data
- **Number of training examples:** 100,000 or fewer
- **Maximum tokens per example:** 2048

### Validation Data
- **Number of validation examples:** 100 or fewer
- **Maximum tokens per example:** 2048

## Requirements
- **Base Model:** You will fine-tune the Llama-3.2-1B model.
- **Training Script:** You must only modify and submit the following two files:
  1. `sft_config.yaml`: This file allows you to adjust the hyperparameters for fine-tuning. 
  2. `sft_utils.py`: This file handles data loading and formatting.
  
  Other files in the project are pre-defined, and you do not need to modify or submit them.

## Download
Please download the required model and dataset from the [Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/cf24e3fa8fbf41e4b0c4/).
After downloading, modify the paths in the sft_config.yaml file accordingly.

## Testing
After you complete the fine-tuning process, we will re-train the model based on the configurations specified in the `sft_config.yaml` file using the following command:
```bash
accelerate launch --config_file=deepspeed_zero3.yaml sft.py
```

The training will use the dataset you **submit** in the required format. You will be responsible for ensuring that your dataset is correctly loaded and formatted, as per the data loading process defined in the `sft_utils.py` file.

This will generate the model checkpoints in the `checkpoints` directory.

The test procedure involves comparing the model’s generated output with the correct answer. Specifically:
- The model will generate an answer in the format `\boxed{answer}` at the end of its output.
- Your solution will be considered correct if the last value inside the `\boxed{}` matches the correct answer.

Your grade will depend on **accuracy**, which is determined by the number of correctly generated answers (where the last `\boxed{}` value matches the correct result). A higher accuracy score will result in a higher grade.

## Submission Instructions
- Submit the modified `sft_config.yaml` and `sft_utils.py` files.
- Submit your own datasets which can be loaded correctly.

## Key Points
- **Modify** only the configuration and utility files.
- **Accuracy** will be your final grade determinant.

Good luck, and happy coding!
