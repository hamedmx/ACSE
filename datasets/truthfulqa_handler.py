from typing import List, Dict, Any
from datasets import load_dataset
from .base_handler import BaseDatasetHandler

class TruthfulQAHandler(BaseDatasetHandler):
    """
    Handler for the TruthfulQA dataset.

    This class ingests the TruthfulQA benchmark. Unlike standard QA, TruthfulQA requires robust 
    semantic matching because the 'correct' answers are open-ended descriptions of the truth. 

    The processed data is subsequently partitioned into disjoint Calibration (D_cal) 
    and Test (D_test) sets.
    """

    def __init__(self, split: str = "validation", seed: int = 42):
        self.split = split
        super().__init__(dataset_name="truthful_qa", seed=seed)

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads the raw TruthfulQA dataset and maps it to the ACSE schema.

        Returns:
            List[Dict]: A list of standardized prompt objects:
                {
                    'id': str,              # Synthesized unique ID (truthfulqa_{index})
                    'prompt': str,          # The input x (the trick question)
                    'reference_answers': List[str] # Valid truths for computing e(x,y)
                }
        """
        dataset = load_dataset("truthful_qa", "generation", split=self.split)
        standardized_data = []
        for i, entry in enumerate(dataset):
            q_id = f"truthfulqa_{i}"
            question_text = entry['question']
            references = entry['correct_answers']
            standardized_entry = {
                'id': q_id,
                'prompt': question_text,
                'reference_answers': references
            }
            
            standardized_data.append(standardized_entry)
            
        return standardized_data