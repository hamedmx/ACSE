from typing import List, Dict, Any
from datasets import load_dataset
from .base_handler import BaseDatasetHandler

class CoQAHandler(BaseDatasetHandler):
    """
    Handler for the CoQA (Conversational Question Answering).

    This class ingests the CoQA benchmark and flattens the conversational 
    turns into independent prompt-reference pairs.

    The flattened data is subsequently partitioned into disjoint Calibration (D_cal) 
    and Test (D_test) sets by the parent class.
    """

    def __init__(self, split: str = "validation", seed: int = 42):
        self.split = split
        super().__init__(dataset_name="coqa", seed=seed)

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads the raw CoQA dataset.

        CoQA is structured as stories containing lists of questions and answers.
        This method iterates through every story and every turn within that story
        to create individual data points.

        Returns:
            List[Dict]: A list of standardized prompt objects:
                {
                    'id': str,              # Composite ID (story_id + turn_index)
                    'prompt': str,          # The input x (the question)
                    'reference_answers': List[str] # All valid ground truths for e(x,y)
                }
        """

        dataset = load_dataset("coqa", split=self.split)
        standardized_data = []
        for entry in dataset:
            story_id = entry['id']
            questions = entry['questions']
            answers_block = entry['answers']
            
            # However, the structure of CoQA in 'datasets' separates answers by turn index.
            # We iterate by index 'i' to align questions with their specific answer set.
            for i, question_text in enumerate(questions):
                
                # Construct a unique ID for this specific prompt-response pair
                # This ensures we can track specific turns during evaluation
                unique_id = f"{story_id}_{i}"
                
                # CoQA often provides a single primary answer in the main fields,
                # but for robustness, we treat it as a list to support the semantic matching
                raw_ref = answers_block['input_text'][i]
                
                # We normalize this into a list of strings to satisfy the reference_answers.
                references = [raw_ref]
                standardized_entry = {
                    'id': unique_id,
                    'prompt': question_text,
                    'reference_answers': references
                }

                standardized_data.append(standardized_entry)

        return standardized_data