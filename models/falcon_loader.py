import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_loader import BaseLLMLoader


class FalconLoader(BaseLLMLoader):
    """
    Unified Falcon Loader supporting:
        - Falcon-7B-Instruct
        - Falcon-8B-Instruct

    Supported checkpoints:
        - tiiuae/falcon-7b-instruct
        - tiiuae/falcon-8b-instruct
    """

    SUPPORTED_MODELS = {
        "7b": "tiiuae/falcon-7b-instruct",
        "8b": "tiiuae/falcon-8b-instruct"
    }

    def __init__(
        self,
        model_size: str = "7b",
        device: str = "cuda"
    ):
        """
        Initializes the Falcon loader.

        Args:
            model_size (str):
                Either:
                    - "7b"
                    - "8b"

            device (str):
                Execution device.
        """

        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Falcon model size: {model_size}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_size = model_size
        model_name = self.SUPPORTED_MODELS[model_size]

        super().__init__(model_name, device)

    def _load_model(self):
        """
        Loads Falcon tokenizer + model.

        Falcon models require:
            trust_remote_code=True
        """

        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

    def _generate_batch(
        self,
        prompt: str,
        n: int,
        top_p: float,
        temperature: float
    ) -> List[str]:
        """
        Generates n stochastic responses for a prompt.

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

        formatted_prompt = f"User: {prompt}\nAssistant:"

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
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        responses = []

        for text in generated_texts:

            if "Assistant:" in text:
                response_only = text.split("Assistant:")[-1].strip()

            else:
                response_only = text.replace(
                    formatted_prompt,
                    ""
                ).strip()

            responses.append(response_only)

        return responses
