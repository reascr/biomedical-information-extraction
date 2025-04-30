from config import MODEL_CONFIGS
from utils import set_seed
from datasets import create_dataloaders
from utils import train_and_evaluate_NER_optuna
from transformers import AutoTokenizer
import optuna
import json
import os
import torch


script_dir = os.path.dirname(os.path.abspath(__file__))
OPTUNA_RESULTS_DIR = os.path.join(script_dir, "..", "results", "NER", "optuna")
os.makedirs(OPTUNA_RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")

DRIVE_UCLOUD_DIR_OPTUNA = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "NER", "optuna"))
os.makedirs(DRIVE_UCLOUD_DIR_OPTUNA, exist_ok=True)

num_labels = 27
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(42)

results = {}

# code modified after https://optuna.org/#code_examples, assessed Apr 11, 2025 (and https://optuna.readthedocs.io/, assessed Apr 11, 2025)
def objective(trial, model_name):
    # take hyperparameter ranges from Gu et al. (2021) for learning rate, batch size, and set dropout minimum value to the value suggested by Gu et al. (2021): 0.1
    lr = trial.suggest_categorical('LR', [1e-5,3e-5, 5e-5]) 
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1) # same for weight decay
    batch_size = trial.suggest_categorical('BATCH_SIZE', [16, 32]) # these are categorical values
    num_epochs = trial.suggest_int('NUM_EPOCHS', 3, 7)
    dropout = trial.suggest_float('dropout', 0.1, 0.3) # uniform sampling on linear scale. 
    max_norm = trial.suggest_float('max_norm', 0.5, 2.0)

    print(f"Trial for {model_name} with parameters:", trial.params)

    # create data loaders for the model
    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size, tokenizer, device)

    # train and evaluate the model
    model, test_micro_f1, test_macro_f1, _, _, _, _ = train_and_evaluate_NER_optuna(
        model_name=model_name,
        seed=42, # to ensure the differences in the scores are due to hyperparamaters and to ensure reproducibility
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        dropout=dropout,
        num_labels=num_labels,
        device=device,
        max_norm=max_norm, track_wandb = False
    )

    return test_micro_f1 # optimize for test micro f1 

# run Optuna for each model and save results
for model_name in MODEL_CONFIGS.keys():
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    study = optuna.create_study(direction='maximize') # we want to maximize micro f1 on test set
    study.optimize(lambda trial: objective(trial, model_name), n_trials=20)

    print(f"Best hyperparameters for {model_name}: {study.best_params}")
    
    # save the best hyperparameters for each model
    results[model_name] = {
        "best_params": study.best_params,
        "best_score": study.best_value
    }


for model_name, model_results in results.items():
    results_file = os.path.join(DRIVE_UCLOUD_DIR_OPTUNA, f"optuna_results_{model_name}.json")
    with open(results_file, "w") as f:
        json.dump(model_results, f, indent=4)
    results_file = os.path.join(OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")
    with open(results_file, "w") as f:
        json.dump(model_results, f, indent=4)
    


###### References ###########

# Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., ... & Poon, H. 2021. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH), 3(1), 1-23.
# Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.
