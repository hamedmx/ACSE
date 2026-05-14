import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class BaseDatasetHandler(ABC):
    """
    Abstract Base Class for Dataset Ingestion.
    
    This class enforces the strict separation of the dataset into Calibration (D_cal) 
    and Test (D_test) sets to ensure the exchangeability assumption holds for 
    Conformal Prediction guarantees.
    
    Structure:
    - Input: Raw dataset name (e.g., 'trivia_qa').
    - Output: Standardized list of dictionaries with keys:
        - 'id': Unique identifier for the prompt (x).
        - 'prompt': The input query string (x).
        - 'reference_answers': List of valid ground-truth answers for semantic scoring e(x, y).
    """

    def __init__(self, dataset_name: str, seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.raw_data: List[Dict[str, Any]] = self._load_data()

    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Abstract method to load and standardize the raw dataset.
        
        Must be implemented by subclasses (TriviaQAHandler, CoQAHandler, etc.)
        to map specific dataset fields (e.g., 'question', 'answers') into the
        standard schema required by the ACSE pipeline.
        
        Returns:
            List[Dict]: The full dataset prompt-reference pairs.
        """
        pass

    def get_splits(self, n_cal: int = 1300, n_test: int = 700) -> Tuple[List[Dict], List[Dict]]:
        """
        Partitions the dataset into disjoint Calibration (D_cal) and Test (D_test).
        
        This implements the "randomly split" logic described in the experimental setup.
        We fix the sample sizes to N_cal = 1300 and N_test = 700 as requested.
        
        Args:
            n_cal (int): Size of the calibration set (D_cal). Default: 1300.
            n_test (int): Size of the test set (D_test). Default: 700.
            
        Returns:
            Tuple[List, List]: (calibration_data, test_data)
        """

        total_required = n_cal + n_test
        if len(self.raw_data) < total_required:
            raise ValueError(f"Dataset size ({len(self.raw_data)}) is smaller than required split ({total_required}).")

        indices = np.arange(len(self.raw_data))
        self.rng.shuffle(indices)
        cal_indices = indices[:n_cal]
        test_indices = indices[n_cal : n_cal + n_test]
        d_cal = [self.raw_data[i] for i in cal_indices]
        d_test = [self.raw_data[i] for i in test_indices]
        return d_cal, d_test