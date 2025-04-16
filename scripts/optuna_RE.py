from config import MODEL_CONFIGS
from datasets import create_dataloaders_RE
from utils import train_and_evaluate_RE, set_seed
from transformers import AutoTokenizer
import optuna
import json
import os
import torch


script_dir = os.path.dirname(os.path.abspath(__file__))
OPTUNA_RESULTS_DIR = os.path.join(script_dir, "..", "results", "RE", "optuna")
os.makedirs(OPTUNA_RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 6
threshold = 0.6
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
    dropout = trial.suggest_uniform('dropout', 0.1, 0.3) # uniform sampling on linear scale. 
    max_norm = trial.suggest_uniform('max_norm', 0.5, 2.0)

    print(f"Trial for {model_name} with parameters:", trial.params)

    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    tokenizer.add_tokens(special_tokens)
    tokenizer_voc_size = len(tokenizer)
    vocab = tokenizer.get_vocab()

    last_four_token_ids = sorted(vocab.values())[-4:] # extract the last four token ids (=entity markers)
    ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id = last_four_token_ids

    # create data loaders for the model
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders_RE(batch_size, tokenizer, device)

    # train and evaluate the model
    model, test_micro_f1, test_macro_f1, _, _, _, _ , _, _= train_and_evaluate_RE(
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
        threshold=threshold,
        track_wandb = False
    )

    return test_micro_f1 # optimize for test micro f1 

# run Optuna for each model and save results
for model_name in MODEL_CONFIGS.keys():
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    study = optuna.create_study(direction='maximize') # we want to maximize micro f1 on test set
    study.optimize(lambda trial: objective(trial, model_name), n_trials=1) # change to 20... or RE 10

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


###### References ###########

# Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., ... & Poon, H. 2021. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH), 3(1), 1-23.
# Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.