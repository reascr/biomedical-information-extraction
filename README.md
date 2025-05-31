# Biomedical Information Extraction
## Investigating the Relationship Between Microbiota and Neurological Disorders in the Gut-Brain Axis

This repository contains code developed for my masters thesis in IT and Cognition at the University of Copenhagen, building a Named Entity Recognition (NER) and Relation Extraction (RE) pipeline for biomedical abstracts using the GutBrainIE2025 dataset (https://hereditary.dei.unipd.it/challenges/gutbrainie/2025/). 

#### Repository Structure

```
.
├── notebooks
│   ├── BiLSTM_CRF_baseline.ipynb
│   ├── ner-statistics.ipynb
│   └── dataset-wordclouds-gutbrainie.ipynb
├── requirements.txt
├── run_on_ucloud.sh
├── scripts
│   ├── config.py
│   ├── datasets.py
│   ├── evaluation_gutbrainie2025.py
│   ├── get_predictions_NER.py
│   ├── get_predictions_RE.py
│   ├── id_rsa
│   ├── models.py
│   ├── optuna_NER.py
│   ├── optuna_RE.py
│   ├── optuna_RE_ternary.py
│   ├── train_evaluate_NER_BERT_models.py
│   ├── train_evaluate_RE_BERT_models.py
│   ├── train_evaluate_ternary_RE_BERT_models.py
│   └── utils.py
```

#### Usage 
To obtain the optimal hyperparameters, run optuna_NER.py, optuna_RE.py (for binary classification RE), and optuna_RE_ternary.py (for multiclass RE).
To fine-tune the models, run the pipeline, generate Named Entity predictions running train_evaluate_NER_BERT_models.py, and then either train_evaluate_RE_BERT_models.py (for binary classification RE) or train_evaluate_ternary_RE_BERT_models.py (for multiclass classification). Make sure to install the dependencies and change all folder paths. Evaluation is conducted within these scripts based on the [evaluation script](https://github.com/MMartinelli-hub/GutBrainIE_2025_Baseline/blob/main/Eval/evaluate.py) of the GutBrainIE2025 challenge.

### Additional Information
The notebooks folder contains the script to run the BiLSTM-CRF tagger (https://github.com/reascr/BiLSTM-CRF-NER-Baseline, forked from https://github.com/glample/tagger). Further, it includes a notebook  to compute the Wilcoxon-signed rank test for statistical significance testing as well as a notebook to generate wordclouds for specific entity labels in the dataset.
