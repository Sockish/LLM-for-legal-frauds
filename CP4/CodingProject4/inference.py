import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class LegalInference:
    def __init__(self, model_path, device="auto"):
        """
        Initialize the inference class with your fine-tuned model
        
        Args:
            model_path: Path to your saved model checkpoint
            device: Device to run inference on ("auto", "cuda", "cpu")
        """
        print(f"Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def create_chat_prompt(self, question, system_prompt=None):
        """
        Create a chat prompt in the same format used during training
        """
        if system_prompt is None:
            system_prompt = "你是一个专业的法律顾问，能够准确分析法律案例并提供详细的解答。"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    
    def generate_response(self, question, system_prompt=None, max_length=2048, temperature=0.7, top_p=0.9):
        """
        Generate response for a legal question
        
        Args:
            question: The legal question to answer
            system_prompt: Custom system prompt (optional)
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        # Create the prompt
        prompt = self.create_chat_prompt(question, system_prompt)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the response (excluding the input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def batch_inference(self, questions, system_prompt=None, **generation_kwargs):
        """
        Perform inference on a batch of questions
        """
        results = []
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}")
            response = self.generate_response(question, system_prompt, **generation_kwargs)
            results.append({
                "question": question,
                "answer": response
            })
        return results

def main():
    # =============================================================================
    # CONFIGURATION - MODIFY THESE SETTINGS
    # =============================================================================
    
    # Model configuration
    model_path = "/root/LLM-for-legal-frauds/CP4/CodingProject4/checkpoints/model_2025_05_30_091731"  # UPDATE WITH YOUR MODEL PATH
    
    # Inference configuration
    config = {
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Choose inference mode:
    # 1. Single question
    # 2. Batch inference from file
    # 3. Interactive mode
    
    # Mode 1: Single question
    single_question = "什么是合同诈骗罪的构成要件？"
    
    # Mode 2: Batch inference
    questions_file = "test_questions.json"  # Path to your questions file
    output_file = "inference_results.json"  # Output file for results
    
    # Mode 3: Interactive mode - set to True to enable
    interactive_mode = False
    
    # Choose which mode to run (comment/uncomment as needed)
    mode = "single"  # Options: "single", "batch", "interactive"
    
    # =============================================================================
    # EXECUTION - NO NEED TO MODIFY BELOW THIS LINE
    # =============================================================================
    
    # Initialize inference
    inferencer = LegalInference(model_path)
    
    if mode == "single":
        # Single question inference
        print(f"\nQuestion: {single_question}")
        print("=" * 50)
        response = inferencer.generate_response(single_question, **config)
        print(f"Answer: {response}")
        
    elif mode == "batch":
        # Batch inference
        print(f"Loading questions from: {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions (handle different formats)
        if isinstance(data, list):
            if isinstance(data[0], dict):
                if "question" in data[0]:
                    questions = [item["question"] for item in data]
                elif "instruction" in data[0]:
                    questions = [item["instruction"] for item in data]
                else:
                    questions = [str(item) for item in data]
            else:
                questions = data
        else:
            raise ValueError("Unsupported questions file format")
        
        # Perform batch inference
        results = inferencer.batch_inference(questions, **config)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    elif mode == "interactive":
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            question = input("\nEnter your legal question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                print("Generating response...")
                response = inferencer.generate_response(question, **config)
                print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()