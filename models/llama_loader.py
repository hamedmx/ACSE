import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_loader import BaseLLMLoader


class LlamaLoader(BaseLLMLoader):
    """
    Unified Llama Loader supporting:

        - Llama-2-7B-Chat
        - Llama-3-8B-Instruct

    Supported checkpoints:
        - meta-llama/Llama-2-7b-chat-hf
        - meta-llama/Meta-Llama-3-8B-Instruct
    """

    SUPPORTED_MODELS = {
        "7b": "meta-llama/Llama-2-7b-chat-hf",
        "8b": "meta-llama/Meta-Llama-3-8B-Instruct"
    }

    def __init__(
        self,
        model_size: str = "7b",
        device: str = "cuda"
    ):
        """
        Initializes the Llama loader.

        Args:
            model_size (str):
                Either:
                    - "7b"  -> Llama-2-7B-Chat
                    - "8b"  -> Llama-3-8B-Instruct

            device (str):
                Execution device.
        """

        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Llama model size: {model_size}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_size = model_size
        model_name = self.SUPPORTED_MODELS[model_size]

        super().__init__(model_name, device)

    def _load_model(self):
        """
        Loads tokenizer + model.
        """

        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:

            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False
        )

        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        """
        Formats prompts according to model family.
        """

        # Llama-2 chat format
        if self.model_size == "7b":
            return f"[INST] {prompt} [/INST]"

        # Llama-3 instruct format
        elif self.model_size == "8b":
            return (
                "<|begin_of_text|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        return prompt

    def _extract_response(
        self,
        generated_text: str,
        formatted_prompt: str
    ) -> str:
        """
        Extracts assistant response from generated text.
        """

        # Llama-2 extraction
        if self.model_size == "7b":

            if "[/INST]" in generated_text:
                return generated_text.split("[/INST]")[-1].strip()

        # Llama-3 extraction
        elif self.model_size == "8b":

            assistant_tag = (
                "<|start_header_id|>assistant<|end_header_id|>"
            )

            if assistant_tag in generated_text:
                return generated_text.split(assistant_tag)[-1] \
                    .replace("<|eot_id|>", "") \
                    .strip()

        return generated_text.replace(
            formatted_prompt,
            ""
        ).strip()

    def _generate_batch(
        self,
        prompt: str,
        n: int,
        top_p: float,
        temperature: float
    ) -> List[str]:
        """
        Generates n stochastic responses for prompt x.

        Args:
            prompt (str):
                Input query x.

            n (int):
                Number of responses to sample.

            top_p (float):
                Nucleus sampling threshold.

            temperature (float):
                Sampling temperature.

        Returns:
            List[str]:
                Generated response set Y(x).
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

            response_only = self._extract_response(
                text,
                formatted_prompt
            )

            responses.append(response_only)

        return responses
