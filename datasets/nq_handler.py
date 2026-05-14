from typing import List, Dict, Any
from datasets import load_dataset
from .base_handler import BaseDatasetHandler

class NQHandler(BaseDatasetHandler):
    """
    Handler for the Natural Questions (NQ) dataset.

    This class ingests the NQ benchmark.
    We retain only those  questions that have at least one valid 
    'short answer' (e.g., an entity, date, or short phrase).

    Questions with long explanations only or unanswerable
    are excluded, as they are less compatible with the precise semantic 
    correctness labeling function e(x, y) used in the ACSE experiments.

    The processed data is subsequently partitioned into disjoint Calibration (D_cal) 
    and Test (D_test) sets.
    """

    def __init__(self, split: str = "validation", seed: int = 42):
        self.split = split
        super().__init__(dataset_name="natural_questions", seed=seed)

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads the raw Natural Questions dataset and filters for valid short answers.

        Returns:
            List[Dict]: A list of standardized prompt objects:
                {
                    'id': str,              # Unique identifier from NQ
                    'prompt': str,          # The input x (the question)
                    'reference_answers': List[str] # Valid short answers for computing e(x,y)
                }
        """
        dataset = load_dataset("natural_questions", split=self.split)
        standardized_data = []
        for entry in dataset:
            q_id = entry['id']
            question_text = entry['question']['text']
            annotations = entry['annotations']
            valid_answers = set()

            for annotation in annotations:
                short_answers = annotation['short_answers']
                if short_answers:
                    for sa in short_answers:
                        if sa['text']:
                            valid_answers.add(sa['text'])
            
            if len(valid_answers) > 0:
                standardized_entry = {
                    'id': q_id,
                    'prompt': question_text,
                    'reference_answers': list(valid_answers)
                }
                
                standardized_data.append(standardized_entry)
                
        return standardized_data