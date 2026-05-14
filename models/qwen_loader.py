import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_loader import BaseLLMLoader

class QwenLoader(BaseLLMLoader):
    """
    Implementation of the LLM Loader for Qwen-7B-Chat.
    """

    def __init__(self, 
                 model_name: str = "Qwen/Qwen-7B-Chat", 
                 device: str = 'cuda'):
        """
        Initializes the Qwen loader.
        
        Args:
            model_name (str): 'Qwen/Qwen-7B-Chat'.
            device (str): Execution device.
        """
        super().__init__(model_name, device)

    def _load_model(self):
        """
        Loads the Qwen-7B-Chat model and tokenizer from Hugging Face.
        """
        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eod_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        self.model.eval()

    def _generate_batch(self, prompt: str, n: int, top_p: float, temperature: float) -> List[str]:
        """
        Executes the stochastic generation process for a single prompt x to produce 
        the response set Y(x) = {y_1, ..., y_n}.

        Args:
            prompt (str): The raw input query x.
            n (int): Number of responses to sample.
            top_p (float): Nucleus threshold eta.
            temperature (float): Softmax temp T.

        Returns:
            List[str]: The set of n generated response strings.
        """
        formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        input_ids = inputs.input_ids.repeat(n, 1)
        attention_mask = inputs.attention_mask.repeat(n, 1)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,          
                top_p=top_p,            
                temperature=temperature,
                max_new_tokens=200,      
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_texts = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=False 
        )

        responses = []
        for text in generated_texts:
            if "<|im_start|>assistant" in text:
                response_part = text.split("<|im_start|>assistant")[-1]
                response_only = response_part.replace("<|im_end|>", "").strip()
            else:
                response_only = text.replace(formatted_prompt, "").strip()
            
            responses.append(response_only)
            
        return responses