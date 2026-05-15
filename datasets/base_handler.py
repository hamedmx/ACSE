import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any


class BaseDatasetHandler(ABC):
    """
    Abstract Base Class for Dataset Ingestion.

    This class standardizes multiple task formats into a unified schema
    for the ACSE pipeline, including:

    - Open-ended QA (TruthfulQA, TriviaQA, CoQA)
    - Summarization (SAMSum)
    - Multiple-choice QA (MMLU)

    Standardized schema:
        {
            'id': str,
            'prompt': str,
            'reference_answers': List[str],

            # Optional task-specific fields
            'choices': List[str],        # For MCQ tasks like MMLU
            'task_type': str             # 'qa', 'summarization', 'mcq'
        }

    The dataset is partitioned into disjoint:
        - Calibration set (D_cal)
        - Test set (D_test)

    to preserve exchangeability assumptions required for
    conformal prediction guarantees.
    """

    def __init__(self, dataset_name: str, seed: int = 42):
        self.dataset_name = dataset_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Full standardized dataset
        self.raw_data: List[Dict[str, Any]] = self._load_data()

    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads and standardizes a dataset into the ACSE schema.

        Returns:
            List[Dict]: Standardized dataset entries.
        """
        pass

    def get_splits(
        self,
        n_train: int = 1200,
        n_cal: int = 400,
        n_test: int = 400
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Randomly partitions the dataset into:
            - Training set
            - Calibration set (D_cal)
            - Test set (D_test)

        Args:
            n_train (int): Number of training samples.
            n_cal (int): Number of calibration samples.
            n_test (int): Number of test samples.

        Returns:
            Tuple[List, List, List]:
                (train_data, calibration_data, test_data)
        """

        total_required = n_train + n_cal + n_test

        if len(self.raw_data) < total_required:
            raise ValueError(
                f"Dataset size ({len(self.raw_data)}) "
                f"is smaller than required split ({total_required})."
            )

        indices = np.arange(len(self.raw_data))
        self.rng.shuffle(indices)

        train_indices = indices[:n_train]

        cal_start = n_train
        cal_end = cal_start + n_cal
        cal_indices = indices[cal_start:cal_end]

        test_start = cal_end
        test_end = test_start + n_test
        test_indices = indices[test_start:test_end]

        d_train = [self.raw_data[i] for i in train_indices]
        d_cal = [self.raw_data[i] for i in cal_indices]
        d_test = [self.raw_data[i] for i in test_indices]

        return d_train, d_cal, d_test

    def dataset_statistics(self) -> Dict[str, Any]:
        """
        Computes basic dataset statistics.

        Returns:
            Dict[str, Any]
        """

        task_types = {}

        for sample in self.raw_data:
            task = sample.get("task_type", "unknown")
            task_types[task] = task_types.get(task, 0) + 1

        return {
            "dataset_name": self.dataset_name,
            "num_samples": len(self.raw_data),
            "task_distribution": task_types,
            "seed": self.seed
        }
