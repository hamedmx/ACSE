from typing import List, Dict, Any
from datasets import load_dataset
from .base_handler import BaseDatasetHandler

class TriviaQAHandler(BaseDatasetHandler):
    """
    Handler for the TriviaQA dataset.
    
    This class is responsible for ingesting the raw TriviaQA data using
     the 'rc.nocontext' configuration.
    
    The loaded data is subsequently partitioned by the parent class 
    into calibration (D_cal) and test (D_test) sets.
    """

    def __init__(self, split: str = "validation", seed: int = 42):
        self.split = split
        super().__init__(dataset_name="trivia_qa", seed=seed)

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads the raw TriviaQA dataset and maps it to the ACSE schema.
        
        Returns:
            List[Dict]: A list of standardized prompt objects:
                {
                    'id': str,              # Unique identifier (question_id)
                    'prompt': str,          # The input x (question)
                    'reference_answers': List[str] # Valid ground truths for computing e(x,y)
                }
        """
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=self.split)
        standardized_data = []

        # Iterate over each example in the raw dataset
        for entry in dataset:
            q_id = entry['question_id']
            question_text = entry['question']
            answer_dict = entry['answer']
            
            # We need the list of all valid answers to determine correctness e(x, y).
            # TriviaQA provides 'aliases' which covers synonyms and variations.
            # If 'aliases' is empty, we fall back to the 'normalized_value'.
            references = answer_dict.get('aliases', [])
            if not references:
                references = [answer_dict['normalized_value']]
            
            standardized_entry = {
                'id': q_id,
                'prompt': question_text,
                'reference_answers': references
            }
            
            standardized_data.append(standardized_entry)

        return standardized_data