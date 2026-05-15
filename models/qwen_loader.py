import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_loader import BaseLLMLoader


class QwenLoader(BaseLLMLoader):
    """
    Unified Qwen Loader supporting:
        - Qwen-7B-Chat
        - Qwen-8B-Chat (or larger compatible chat variants)

    This class standardizes prompt formatting and extraction across Qwen model sizes.
    """

    SUPPORTED_MODELS = {
        "7b": "Qwen/Qwen-7B-Chat",
        "8b": "Qwen/Qwen-8B-Chat"
    }

    def __init__(
        self,
        model_size: str = "7b",
        device: str = "cuda"
    ):
        """
        Args:
            model_size (str): "7b" or "8b"
            device (str): execution device
        """

        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Qwen model size: {model_size}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_size = model_size
        model_name = self.SUPPORTED_MODELS[model_size]

        super().__init__(model_name, device)

    def _load_model(self):
        """
        Loads Qwen tokenizer and model.
        """

        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        # Qwen token handling
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif hasattr(self.tokenizer, "eod_id"):
                self.tokenizer.pad_token = self.tokenizer.eod_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        """
        Standardizes chat formatting across Qwen versions.
        """

        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _extract_response(self, text: str, formatted_prompt: str) -> str:
        """
        Extract assistant response robustly across model variants.
        """

        if "<|im_start|>assistant" in text:
            response_part = text.split("<|im_start|>assistant")[-1]
            return response_part.replace("<|im_end|>", "").strip()

        return text.replace(formatted_prompt, "").strip()

    def _generate_batch(
        self,
        prompt: str,
        n: int,
        top_p: float,
        temperature: float
    ) -> List[str]:
        """
        Generates n stochastic responses for a prompt.
        """

        formatted_prompt = self._format_prompt(prompt)

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
            responses.append(
                self._extract_response(text, formatted_prompt)
            )

        return responses
