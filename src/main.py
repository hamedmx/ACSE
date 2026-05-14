import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import util
from pipeline import SemanticClusterer, AdaptiveUncertaintyEngine, ConformalEngine
from utils import ACSEMetrics, ACSEPlotter, AblationSuite
from datasets.triviaqa_handler import TriviaQAHandler
from models.mistral_loader import MistralLoader

class ACSEPipeline:
    """
    Main Orchestrator for the ACSE Framework.
    """

    def __init__(self, args):
        self.args = args
        
        # 1. Initialize Dataset Handler 
        if args.dataset == "trivia_qa":
            self.handler = TriviaQAHandler(seed=args.seed)
        else:
            raise ValueError(f"Handler for {args.dataset} not implemented.")

        # 2. Initialize LLM Loader 
        self.llm = MistralLoader(model_name=args.model_name, device=args.device)
        
        # 3. Initialize Core ACSE Components
        self.clusterer = SemanticClusterer(epsilon=0.35) 
        self.uncertainty_engine = AdaptiveUncertaintyEngine() 
        self.conformal_engine = ConformalEngine(alpha=args.alpha)
        self.plotter = ACSEPlotter()
        self.labeling_model = self.clusterer.encoder
        self.cosine_threshold = 0.7 

    def compute_correctness_labels(self, responses: List[str], references: List[str]) -> List[int]:
        """
        Computes binary error labels e(x, y) via semantic matching. 
        """
        resp_embs = self.labeling_model.encode(responses, convert_to_tensor=True)
        ref_embs = self.labeling_model.encode(references, convert_to_tensor=True)
        sim_matrix = util.cos_sim(resp_embs, ref_embs).cpu().numpy()
        max_sims = np.max(sim_matrix, axis=1)
        return (max_sims < self.cosine_threshold).astype(int).tolist()

    def run_stage(self, dataset: List[Dict], stage_name: str) -> List[Dict]:
        """
        Executes the processing loop: Sampling -> Clustering -> Labeling. 
        """
        results = []
        print(f"\n--- Processing {stage_name} Data ({len(dataset)} samples) ---")
        
        for item in tqdm(dataset):
            responses = self.llm.get_response_set(item['prompt'])
            clustering_data = self.clusterer.process_prompt(responses)
            resp_error_labels = self.compute_correctness_labels(responses, item['reference_answers'])
            k_star = np.argmax(clustering_data['probs'])
            i_star = np.argmax(clustering_data['soft_assignments'][:, k_star])
            prompt_error = resp_error_labels[i_star]
            
            results.append({
                **item,
                'responses': responses,
                'clustering_data': clustering_data,
                'error_labels_response': resp_error_labels,
                'error_label_prompt': prompt_error
            })
        return results

    def run(self):
        cal_raw, test_raw = self.handler.get_splits(n_cal=1300, n_test=700)
        cal_results = self.run_stage(cal_raw, "Calibration")
        print("\n[Calibration] Fitting distributional constants...")
        self.uncertainty_engine.fit_calibration_stats([r['clustering_data'] for r in cal_results])

        cal_processed = []
        for r in cal_results:
            out = self.uncertainty_engine.compute_inflated_score(r['clustering_data'])
            cal_processed.append({**r, **out})
            
        print(f"\n[Calibration] Learning thresholds (alpha={self.args.alpha})...")
        self.conformal_engine.calibrate(cal_processed)
        print(f"   > tau_hat: {self.conformal_engine.tau_hat:.4f}")
        print(f"   > q_hat:   {self.conformal_engine.q_hat:.4f}")
        
        test_results = self.run_stage(test_raw, "Inference")

        preds_sets, decisions, u_hats, labels = [], [], [], []

        print("\n[Inference] Applying Decision Rules and Generating Status...")
        for r in test_results:
            # 1. Calculate inflated uncertainty score for the prompt
            out = self.uncertainty_engine.compute_inflated_score(r['clustering_data'])
    
            # 2. Compare against the conformal threshold (tau_hat) inside the engine
            decision = self.conformal_engine.predict({**r, **out})
    
            # 3. Final decision handling for individual prompts 
            if decision['accepted']:
                # If score <= tau_hat, return representative dominant cluster member
                set_indices = decision['prediction_set_indices']
                best_answer = r['responses'][set_indices[0]] if set_indices else "N/A"
                print(f"ACCEPTED  | Prompt ID: {r['id']} | Predicted Answer: {best_answer}")
            else:
                # If score > tau_hat, the model abstains to maintain risk budget [cite: 513, 658]
                print(f"ABSTAINED | Prompt ID: {r['id']} | Confidence Low: Deferred to prevent hallucination.")

            set_strs = [r['responses'][i] for i in decision['prediction_set_indices']]
            preds_sets.append(set_strs)
            decisions.append(decision['accepted'])
            u_hats.append(out['u_hat'])
            labels.append(r['error_label_prompt'])
            
        # --- 4. Evaluation ---
        print("\n--- Final Performance Metrics ---")
        disc_results = ACSEMetrics.compute_discrimination_metrics(np.array(u_hats), np.array(labels))
        print(f"Hallucination Detection: {disc_results}")
        
        conf_results = ACSEMetrics.compute_conformal_metrics(
            preds_sets, [t['reference_answers'][0] for t in test_raw], 
            decisions, labels, self.args.alpha
        )
        print(f"Conformal Reliability: {conf_results}")
        
        # --- 5. Ablations & Plotting ---
        if self.args.run_ablations:
            print("\n--- Running Analytical Studies ---")
            all_resps = [r['responses'] for r in test_results]
            cluster_outs = [r['clustering_data'] for r in test_results]
            
            AblationSuite.run_sampling_ablation(all_resps, np.array(labels))
            AblationSuite.run_feature_ablation(cluster_outs, np.array(labels))

            base_u = np.array([c['u_x'] for c in cluster_outs])
            AblationSuite.run_scatter_plot_analysis(np.array(u_hats), 1.0 - base_u, np.array(labels))
            self.plotter.plot_scatter_analysis(np.array(u_hats), 1.0 - base_u, np.array(labels), "Base SE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACSE Implementation")
    parser.add_argument("--alpha", type=float, default=0.10, help="Target error tolerance")
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_ablations", action="store_true", help="Generate ablation results and plots")
    
    args = parser.parse_args()
    ACSEPipeline(args).run()