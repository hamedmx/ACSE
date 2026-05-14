from typing import List, Dict, Any
from datasets import load_dataset
from .base_handler import BaseDatasetHandler

class MMLUHandler(BaseDatasetHandler):
    """
    Handler for the MMLU (Massive Multitask Language Understanding) dataset.

    While MMLU is traditionally a multiple-choice classification benchmark, 
    the ACSE framework operates in a generative open-ended setting. 
    Therefore, this handler reconstructs the prompt 'x' to include the options 
    and treats the text of the correct option as the reference 'y' for 
    semantic correctness scoring e(x, y).

    The processed data is subsequently partitioned into disjoint Calibration (D_cal) 
    and Test (D_test) sets..
    """

    def __init__(self, split: str = "test", seed: int = 42):
        self.split = split
        super().__init__(dataset_name="cais/mmlu", seed=seed)

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Loads the raw MMLU dataset.

        Returns:
            List[Dict]: A list of standardized prompt objects:
                {
                    'id': str,              # Synthesized ID (mmlu_{subset}_{index})
                    'prompt': str,          # The formatted input x
                    'reference_answers': List[str] # The text of the correct option
                }
        """
        try:
            dataset = load_dataset("cais/mmlu", "all", split=self.split)
        except:
            dataset = load_dataset("cais/mmlu", "astronomy", split=self.split)

        standardized_data = []
        option_map = ["A", "B", "C", "D"]

        for i, entry in enumerate(dataset):
            q_id = f"mmlu_{i}"
            question_raw = entry['question']
            choices = entry['choices']
            answer_idx = entry['answer'] 

            formatted_options = "\n".join([f"{option_map[k]}) {choice}" for k, choice in enumerate(choices)])
            prompt_text = f"{question_raw}\nOptions:\n{formatted_options}\nAnswer:"

            correct_text = choices[answer_idx]
            references = [correct_text, option_map[answer_idx]]
            
            standardized_entry = {
                'id': q_id,
                'prompt': prompt_text,
                'reference_answers': references
            }
            
            standardized_data.append(standardized_entry)
            
        return standardized_data