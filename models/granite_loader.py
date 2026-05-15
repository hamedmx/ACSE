import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_loader import BaseLLMLoader


class GraniteLoader(BaseLLMLoader):
    """
    Implementation of the LLM Loader for IBM Granite 3.1 8B Instruct.

    This loader standardizes generation across Granite chat/instruct formatting.
    """

    SUPPORTED_MODELS = {
        "8b": "ibm-granite/granite-3.1-8b-instruct"
    }

    def __init__(
        self,
        model_size: str = "8b",
        device: str = "cuda"
    ):
        """
        Args:
            model_size (str): Only "8b" supported for Granite 3.1.
            device (str): Execution device.
        """

        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Granite model size: {model_size}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_size = model_size
        model_name = self.SUPPORTED_MODELS[model_size]

        super().__init__(model_name, device)

    def _load_model(self):
        """
        Loads IBM Granite tokenizer and model.
        """

        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        """
        Granite chat formatting.

        Granite uses a structured instruction style prompt format.
        """

        return (
            "<|system|>\nYou are a helpful assistant.\n"
            "<|user|>\n"
            f"{prompt}\n"
            "<|assistant|>\n"
        )

    def _extract_response(self, text: str, formatted_prompt: str) -> str:
        """
        Extract assistant response robustly.
        """

        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()

        return text.replace(formatted_prompt, "").strip()

    def _generate_batch(
        self,
        prompt: str,
        n: int,
        top_p: float,
        temperature: float
    ) -> List[str]:
        """
        Generates n stochastic responses for a single prompt x.

        Args:
            prompt (str): Input query.
            n (int): Number of samples.
            top_p (float): Nucleus sampling threshold.
            temperature (float): Sampling temperature.

        Returns:
            List[str]: Generated responses.
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
