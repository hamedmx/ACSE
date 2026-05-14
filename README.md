# Adaptive Conformal Semantic Entropy (ACSE)

Adaptive Conformal Semantic Entropy (ACSE) is a method for reliable uncertainty quantification in Large Language Models (LLMs). It estimates prompt-level uncertainty by measuring the semantic dispersion of generated responses and applies conformal prediction to strictly bound the error rate of accepted answers. This repository contains the implementation for the ACSE pipeline including data ingestion, semantic clustering, adaptive inflation and conformal inference.

## Environment Setup
We recommend using a dedicated Python environment to ensure dependency compatibility.
### Prerequisites:
1. Python 3.10+
2. CUDA 11.8+

### Installation:
```bash
# Create and activate environment
conda create -n acse_env python=3.10 -y
conda activate acse_env

# Install dependencies from requirements.txt
pip install -r src/requirements.txt
```
## Usage
The system is orchestrated via `src/main.py`, which handles downloading the data, performing calibration and running inference. To run the full pipeline on TriviaQA with Mistral-7B model, execute the below code:
```bash
python src/main.py \
    --dataset trivia_qa \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --alpha 0.10 \
    --device cuda
```
### Configuration Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `trivia_qa` | Target dataset for evaluation (currently supports TriviaQA). |
| `--model_name` | `mistralai...` | HuggingFace model ID for the backbone LLM. |
| `--alpha` | `0.10` | Target miscoverage level (e.g., 0.10 for 90% reliability). |
| `--seed` | `42` | Random seed for reproducibility. |
| `--device` | `cuda` | Hardware accelerator (`cuda` or `cpu`). |
| `--run_ablations`| `False` | Flag to execute feature ablation studies after inference. |

For a complete run that handles logging as well, use the provided shell script:
```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```
### Modular Usage
  If you prefer to integrate specific ACSE components into your own workflow, you can import them directly from the codebase.

1. **Using the Data Handler**: Load and split the TriviaQA dataset into disjoint calibration and test sets.
```bash
from datasets.triviaqa_handler import TriviaQAHandler

# Initialize handler
handler = TriviaQAHandler()

# Get disjoint splits
cal_data = handler.get_calibration_split()  # N=1300
test_data = handler.get_test_split()        # N=700
```

2. **Running the Model**: Generate responses using the specific nucleus sampling parameters ($\eta=0.9, T=0.35$).
```bash
from models.mistral_handler import MistralLoader

loader = MistralLoader(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"
)

# Generate 10 stochastic samples for a prompt
responses = loader.generate(
    prompt="Who won the 1994 World Cup?",
    num_return_sequences=10
)
```

3. **Calculating Adaptive Entropy**: Compute the inflated uncertainty score û(x) for a set of generated responses.
```bash
from pipeline.inflated_SE import AdaptiveUncertaintyEngine

engine = AdaptiveUncertaintyEngine()

# Calculate score based on semantic dispersion
uncertainty_score = engine.compute_metric(
    responses=responses,
    clustering_threshold=0.35
)
print(f"ACSE Score: {uncertainty_score}")
```

## Notes
1. **HuggingFace Access**: Ensure you have a valid ```HF_TOKEN``` environment variable set for using gated models.
2. **Hardware**: Inference requires approximately 16GB+ VRAM for 7B models in half-precision.
3. **Deterministic Calibration**: While text generation is stochastic, the data splitting and calibration steps are seeded. Always use `--seed` argument (default: 42) to maintain the disjoint train/test splits across different runs.
