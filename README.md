# Counterfactual Cultural Cues Reduce Medical QA Accuracy in LLMs: Identifier vs Context Effects

## Abstract
This repository accompanies our study on the robustness of medical language models to non-decisive cultural information. We introduce a counterfactual benchmark that augments MedQA questions with culture-related identifiers and contextual cues, while preserving the clinically correct diagnosis, to measure whether and how cultural signals influence diagnostic accuracy. We evaluate multiple frontier and open-source medical LLMs under different prompting setups and analyze both accuracy shifts and reasoning failures induced by cultural cues.

## Contribution

- Introduce a counterfactual cultural benchmark with 1,650 variants derived from 150 MedQA items, covering multiple cultural groups and a neutral control.

- Provide a clinician-validated dataset ensuring gold-label invariance across all cultural variants.

- Systematically evaluate GPT-5.2, LLaMA-3.1-8B, DeepSeek-R1, and MedGemma (4B/27B) under option-only and short-explanation prompting.

- Show that cultural identifiers and contextual cues significantly degrade diagnostic accuracy, especially when combined.

- Propose and apply a human-validated LLM-as-judge rubric linking culture-referential reasoning to incorrect diagnoses.

- Release prompts, augmentations, and evaluation scripts to support reproducibility and future mitigation research.
## Evaluation pipeline

![Technology-animation-gif](https://github.com/HIVE-UofT/Evaluatiog-Cultural-Clues-Medical-LLMs/blob/main/Figs/diagram.png)

## Repository Structure

```text
.
├── Code/                                   # Model inference and evaluation code
│   ├── llama_option_diagnose.py            # LLaMA option-based diagnosis inference
│   ├── llama_short_diagnose.py             # LLaMA short-answer diagnosis inference
│   └── ...
├── Data/                                   # Datasets, model outputs, and evaluation results
├── Scripts/                                # Shell scripts for running experiments
├── Figs/                                   # Figures used in the paper
└── README.md                               # Project documentation
```

## Citation

```text
@article{rezaei2026counterfactual,
  title={Counterfactual Cultural Cues Reduce Medical QA Accuracy in LLMs: Identifier vs Context Effects},
  author={Rezaei, Amirhossein Haji Mohammad and Shakeri, Zahra},
  journal={arXiv preprint arXiv:2601.20102},
  year={2026}
}
```
