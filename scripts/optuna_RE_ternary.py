from config import MODEL_CONFIGS
from datasets import create_dataloaders_RE_ternary
from utils import train_and_evaluate_RE_ternary_optuna, set_seed
from transformers import AutoTokenizer
import optuna
import json
import os
import torch
import time
start_time = time.time()


script_dir = os.path.dirname(os.path.abspath(__file__))
OPTUNA_RESULTS_DIR = os.path.join(script_dir, "..", "results", "RE-ternary", "optuna")
os.makedirs(OPTUNA_RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")

DRIVE_UCLOUD_DIR_OPTUNA = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "optuna"))
os.makedirs(DRIVE_UCLOUD_DIR_OPTUNA, exist_ok=True)

print("Saving to:", DRIVE_UCLOUD_DIR_OPTUNA)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 6
threshold = 0.5
special_tokens = ['<ent1>', '</ent1>', '<ent2>', '</ent2>'] # entity markers that will be added to the tokenizers
results = {}

set_seed(42)

# code modified after https://optuna.org/#code_examples, assessed Apr 11, 2025 (and https://optuna.readthedocs.io/, assessed Apr 11, 2025)
def objective(trial, model_name):
    # take hyperparameter ranges from Gu et al. (2021) for learning rate, batch size, and set dropout minimum value to the value suggested by Gu et al. (2021): 0.1
    lr = trial.suggest_categorical('LR', [1e-5,3e-5, 5e-5]) 
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1) 
    batch_size = trial.suggest_categorical('BATCH_SIZE', [16, 32]) # these are categorical values
    # num_epochs = trial.suggest_int('NUM_EPOCHS', 3, 7) # since relation extraction training is time intensive, we'll use 6 epochs with early stopping with patience = 1
    dropout = trial.suggest_float('dropout', 0.1, 0.3) 

    max_norm = trial.suggest_float('max_norm', 0.5, 2.0)

    print(f"Trial for {model_name} with parameters:", trial.params)

    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    tokenizer.add_tokens(special_tokens)
    tokenizer_voc_size = len(tokenizer)
    vocab = tokenizer.get_vocab()

    last_four_token_ids = sorted(vocab.values())[-4:] # extract the last four token ids (=entity markers)
    ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id = last_four_token_ids

    # create data loaders for the model
    train_dataloader, val_dataloader, test_dataloader, num_labels, _ = create_dataloaders_RE_ternary(batch_size, tokenizer, device)

    # train and evaluate the model
    best_val_micro= train_and_evaluate_RE_ternary_optuna(
        model_name=model_name,
        tokenizer_voc_size= tokenizer_voc_size,
        seed=42,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        dropout=dropout,
        device=device,
        max_norm=max_norm,
        ent1_start_id = ent1_start_id, 
        ent1_end_id = ent1_end_id, 
        ent2_start_id = ent2_start_id, 
        ent2_end_id = ent2_end_id, 
        num_labels=num_labels,
        track_wandb = False
    )

    return best_val_micro # optimize for best micro F1 on the validation set

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
    results_file = os.path.join(OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")
    with open(results_file, "w") as f:
        json.dump(model_results, f, indent=4)
    results_file = os.path.join(DRIVE_UCLOUD_DIR_OPTUNA, f"optuna_results_{model_name}.json") # save on server drive
    with open(results_file, "w") as g:
        json.dump(model_results, g, indent=4)


end_time = time.time()
total_time = end_time - start_time

# Format time nicely
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)
formatted_time = f"Total execution time: {int(hours)}h {int(mins)}m {secs:.2f}s"

# Save to a .txt file
time_file_path = os.path.join(DRIVE_UCLOUD_DIR_OPTUNA, "execution_time.txt")
with open(time_file_path, "w") as f:
    f.write(formatted_time)

print(formatted_time)