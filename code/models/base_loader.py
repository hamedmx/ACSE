import torch
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseLLMLoader(ABC):
    """
    Abstract Base Class for Large Language Model Loaders.

    This class establishes the strict contract for response generation.

    """
    N_SAMPLES = 10 
    NUCLEUS_ETA = 0.9 
    TEMPERATURE = 0.3 

    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        Initialize the model loader.
        
        Args:
            model_name (str): The Hugging Face model identifier.
            device (str): Execution device ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Abstract method to load model weights and tokenizer.
        Must be implemented by the specific architecture handler.
        """
        pass

    def get_response_set(self, prompt: str) -> List[str]:
        """
        Generates the set of n responses Y(x) = {y_1, ..., y_n} for a prompt x.
        Args:
            prompt (str): The input prompt x.
            
        Returns:
            List[str]: A list of n generated text responses.
        """
        return self._generate_batch(
            prompt=prompt,
            n=self.N_SAMPLES,
            top_p=self.NUCLEUS_ETA,
            temperature=self.TEMPERATURE
        )

    @abstractmethod
    def _generate_batch(self, prompt: str, n: int, top_p: float, temperature: float) -> List[str]:
        """
        Abstract method to execute the decoding loop.
        
        Args:
            prompt (str): Input text.
            n (int): Number of independent samples to draw.
            top_p (float): The cumulative probability threshold (eta).
            temperature (float): Softmax temperature.
            
        Returns:
            List[str]: The raw text of the generated responses.
        """
        pass