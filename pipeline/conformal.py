import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class ConformalEngine:
    """
    Implements the Conformal Prediction (CP) layer for Adaptive Conformal Semantic Entropy (ACSE).
    
    This engine calibrates two distinct decision boundaries using a hold-out calibration set D_cal:
    1. Prompt-level Acceptance Threshold (tau_hat): Decides whether to answer or abstain.
    2. Response-level Prediction Set Threshold (q_hat): Constructs a set of likely correct responses.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Initialize the conformal engine.
        
        Args:
            alpha (float): The user-specified miscoverage level.
                           Controls the trade-off between abstention and coverage.
        """
        self.alpha = alpha
        # Calibrated thresholds
        self.tau_hat = 1.0  # Prompt-level acceptance threshold 
        self.q_hat = 1.0    # Response-level prediction set threshold 

    def _compute_finite_sample_quantile(self, scores: np.ndarray) -> float:
        """
        Computes the conformal quantile threshold using the finite-sample correction.
        
        Args:
            scores (np.ndarray): Array of calibration non-conformity scores.
            
        Returns:
            float: The calibrated threshold value.
        """
        n_cal = len(scores)
        
        # If calibration set is empty, return max possible score 
        if n_cal == 0:
            return 1.0

        sorted_scores = np.sort(scores)
        k = np.ceil((n_cal + 1) * (1.0 - self.alpha))
        idx = int(k) - 1
  
        # If idx >= n_cal, it implies alpha is too small for the dataset size; we take the max.
        idx = min(max(idx, 0), n_cal - 1)
        return sorted_scores[idx]

    def compute_response_conformity(self, 
                                    s_ik: np.ndarray, 
                                    probs: np.ndarray) -> np.ndarray:
        """
        Computes the Response Conformity Measure phi_i.

        Args:
            s_ik (np.ndarray): Soft assignment matrix of shape.
            probs (np.ndarray): Cluster distribution P(C_k) of shape.
            
        Returns:
            np.ndarray: Array of phi_i scores for each response y_i.
        """
        # Find the best matching cluster index k_hat for each response
        k_hat_indices = np.argmax(s_ik, axis=1)
        row_indices = np.arange(len(s_ik))
        strongest_assignment = s_ik[row_indices, k_hat_indices]
        cluster_support = probs[k_hat_indices]
        phi = strongest_assignment * cluster_support

        return phi

    def compute_response_nonconformity(self, 
                                       u_hat: float, 
                                       phi: np.ndarray) -> np.ndarray:
        """
        Computes the Response-Level Non-Conformity Score S(x, y_i).
        
        Args:
            u_hat (float): Inflated prompt-level uncertainty.
            phi (np.ndarray): Response conformity scores phi_i.
            
        Returns:
            np.ndarray: Array of non-conformity scores in [0, 1].
        """
        atypicality = 1.0 - phi
        S_scores = 0.5 * (u_hat + atypicality)
        
        return S_scores

    def calibrate(self, 
                  calibration_data: List[Dict[str, Any]]):
        """
        Performs Split-Conformal Calibration on D_cal to learn thresholds tau_hat and q_hat.
        
        Args:
            calibration_data: List of dictionaries (one per prompt) containing:
                - 'u_hat': Inflated semantic uncertainty.
                - 'soft_assignments': Matrix s_ik.
                - 'probs': Distribution P(C_k).
                - 'error_label_prompt': Binary label E(x) (0=Correct, 1=Incorrect).
                - 'error_labels_response': List of binary labels e(x, y_i).
        """
        # --- 1. Prompt-Level Calibration ---
        prompt_scores_correct = []
        
        for data in calibration_data:
            # We calibrate on CORRECT prompts to ensure coverage of correctness
            if data['error_label_prompt'] == 0:
                prompt_scores_correct.append(data['u_hat'])
        
        # Compute tau_hat using the finite-sample quantile 
        self.tau_hat = self._compute_finite_sample_quantile(np.array(prompt_scores_correct))
        
        # --- 2. Response-Level Calibration --- 
        response_scores_correct = []
        
        for data in calibration_data:
            u_hat = data['u_hat']
            s_ik = data['soft_assignments']
            probs = data['probs']
            resp_errors = data['error_labels_response'] 
            
            # Compute conformity phi for all responses of this prompt
            phi = self.compute_response_conformity(s_ik, probs)
            
            # Compute non-conformity S for all responses
            S_scores = self.compute_response_nonconformity(u_hat, phi)
            
            # Filter for correct responses only 
            for score, error in zip(S_scores, resp_errors):
                if error == 0:
                    response_scores_correct.append(score)
                    
        # Compute q_hat using the finite-sample quantile 
        self.q_hat = self._compute_finite_sample_quantile(np.array(response_scores_correct))

    def predict(self, 
                inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies Conformal Decision Rules to a new test prompt x_new.
        
        Args:
            inference_data: Dictionary containing u_hat, soft_assignments, probs, etc.
            
        Returns:
            Dict containing:
                - 'decision': "Accept" or "Abstain".
                - 'prediction_set_indices': Indices of responses included in C_alpha.
                - 'S_scores': Raw non-conformity scores for analysis.
        """
        u_hat = inference_data['u_hat']
        s_ik = inference_data['soft_assignments']
        probs = inference_data['probs']
        
        # --- 1. Prompt-Level Decision ---
        is_accepted = (u_hat <= self.tau_hat)
        decision = "Accept" if is_accepted else "Abstain"
        
        # --- 2. Response-Level Prediction Set ---

        phi = self.compute_response_conformity(s_ik, probs)
        S_scores = self.compute_response_nonconformity(u_hat, phi)

        prediction_set_mask = (S_scores <= self.q_hat)
        prediction_set_indices = np.where(prediction_set_mask)[0].tolist()
        
        return {
            'decision': decision,
            'accepted': is_accepted,
            'prediction_set_indices': prediction_set_indices,
            'S_scores': S_scores,
            'tau_hat': self.tau_hat,
            'q_hat': self.q_hat
        }