from typing import List, Dict, Any
from datasets import load_dataset
from .base_handler import BaseDatasetHandler

class SAMSumHandler(BaseDatasetHandler):
    """
    Handler for the SAMSum dataset.

    This class ingests the SAMSum dialogue summarization benchmark.
    Each sample consists of a multi-turn conversation and its
    corresponding human-written summary.

    The processed data is subsequently partitioned into disjoint
    Calibration (D_cal) and Test (D_test) sets.
    """

    def __init__(self, split: str = "test", seed: int = 42):
        self.split = split
        super().__init__(dataset_name="samsum", seed=seed)

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads the raw SAMSum dataset and maps it to the ACSE schema.

        Returns:
            List[Dict]: A list of standardized prompt objects:
                {
                    'id': str,                    # Synthesized unique ID (samsum_{index})
                    'prompt': str,               # The dialogue input x
                    'reference_answers': List[str] # Ground-truth summaries
                }
        """
        dataset = load_dataset("samsum", split=self.split)

        standardized_data = []

        for i, entry in enumerate(dataset):
            sample_id = f"samsum_{i}"

            dialogue_text = entry["dialogue"]
            summary_text = entry["summary"]

            standardized_entry = {
                "id": sample_id,
                "prompt": dialogue_text,
                "reference_answers": [summary_text]
            }

            standardized_data.append(standardized_entry)

        return standardized_data
