from config import MODEL_CONFIGS, label_map, seeds
from datasets import AnnotationDataset, split_datasets, NERDataset, create_dataloaders
from utils import set_seed, plot_train_val_metrics, train_and_evaluate_NER, calculate_averages
from get_predictions_NER import extract_entities, predict_entities
from evaluation_gutbrainie2025 import eval_6_1_NER, eval_submission_6_1_NER, GROUND_TRUTH_PATH, LEGAL_ENTITY_LABELS
from dotenv import load_dotenv
import os
import json
import torch
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(script_dir)
print(f"Current working directory: {os.getcwd()}")

RESULTS_DIR = os.path.join(script_dir,"..", "results", "NER", "BERT-models") 
os.makedirs(RESULTS_DIR, exist_ok=True)
OPTUNA_RESULTS_DIR = os.path.join(script_dir, "..", "results", "NER", "optuna")
print(f"File path: {os.path.abspath(OPTUNA_RESULTS_DIR)}")
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")

load_dotenv()  # load the .env file with the wandb key
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Directory to save the best models (so we can use them for RE)
MODEL_SAVE_DIR = os.path.join("..", "results", "NER", "best-models") 
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

NUM_LABELS = 27
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
best_models = {}
results = []

for model_name in MODEL_CONFIGS.keys():
    file_path = os.path.join(OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        
    best_params = data[model_name]["best_params"] # get the best paramaters from the optuna study

    BATCH_SIZE = best_params["BATCH_SIZE"]
    NUM_EPOCHS = best_params["NUM_EPOCHS"]
    LR = best_params["LR"]
    WEIGHT_DECAY = best_params["weight_decay"]
    DROPOUT = best_params["dropout"]
    MAX_NORM = best_params["max_norm"]

    wandb.init(project="NER", entity="lp2", config={
    "model": model_name,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "dropout": DROPOUT,
    "weight_decay": WEIGHT_DECAY,
    "max_norm": MAX_NORM
    },
    name=f"{model_name}")
    config = wandb.config

    # create dataloaders for each model
    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(BATCH_SIZE,tokenizer, DEVICE)
    test_micro_f1_scores = []
    test_macro_f1_scores = []
    all_train_losses = []
    all_val_losses = []
    all_train_f1s = []
    all_val_f1s = []
    
    best_micro_f1 = -1
    best_macro_f1 = -1
    best_model_state = None
    best_model_seed = None

    #for seed in range(NUM_SEEDS):
    for seed in seeds:
        # train and evaluate model
        model, test_micro_f1, test_macro_f1, train_losses, val_losses, train_f1s, val_f1s = train_and_evaluate_NER(
            model_name, seed, train_dataloader, val_dataloader, test_dataloader, LR, WEIGHT_DECAY, NUM_EPOCHS, DROPOUT, NUM_LABELS, DEVICE
        )
        
        test_micro_f1_scores.append(test_micro_f1)
        test_macro_f1_scores.append(test_macro_f1)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_f1s.append(train_f1s)
        all_val_f1s.append(val_f1s)
        
        # plot the results for this seed
        plot_train_val_metrics(train_losses, val_losses, train_f1s, val_f1s, model_name, seed)

        # save the best model (micro F1)
        if test_micro_f1 > best_micro_f1:
            best_micro_f1 = test_micro_f1
            best_model_state = model.state_dict()
            best_model_seed = seed
        if test_macro_f1 > best_macro_f1:
            best_macro_f1 = test_macro_f1

        seed_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{seed}.pt")
        #torch.save(model.state_dict(), seed_model_path) # outcomment if you want to save the models
        # save the best model for this model_name (we need this later)
        best_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{seed}_best.pt") # because we need to have predictions over all 5 runs for entity level F1 as well
        best_models[model_name] = best_model_path

    # calculate average F1 scores
    mean_micro_f1 = np.mean(test_micro_f1_scores)
    std_micro_f1 = np.std(test_micro_f1_scores)
    mean_macro_f1 = np.mean(test_macro_f1_scores)
    std_macro_f1 = np.std(test_macro_f1_scores)
    
    results.append({
        "model": model_name,
        "avg_test_micro_f1": mean_micro_f1,
        "std_test_micro_f1": std_micro_f1,
        "avg_test_macro_f1": mean_macro_f1,
        "std_test_macro_f1": std_macro_f1,
        "best_micro_f1": best_micro_f1,
        "best_macro_f1": best_macro_f1,
        "best_model_seed": best_model_seed,
        "best_model_path": best_model_path
    })


results_df = pd.DataFrame(results)
results_json_path= os.path.join(RESULTS_DIR, f"results_NER_best_models.json") 
results_df.to_json(results_json_path, orient="records", indent=4)

wandb.finish()



# Plot the average micro and macro F1 for all three models 
pastel_colors = { 
    "BERT": "#D6E8FF",     
    "BioBERT": "#E4D5FF",  
    "PubMedBERT": "#FFCCE5" 
}

model_order = ["BERT", "BioBERT", "PubMedBERT"]
results_df = results_df.set_index('model')  
results_df = results_df.loc[model_order]   
results_df = results_df.reset_index()     


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Micro F1 Score
bars1 = ax1.bar(results_df["model"], results_df["avg_test_micro_f1"], 
                color=[pastel_colors.get(model, "#D6E8FF") for model in results_df["model"]],
                linewidth=1.2, alpha=0.85)

for bar in bars1: # label bars
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", 
             ha="center", va="bottom", fontsize=12, fontweight="bold", color="#444")

ax1.set_ylabel("Average Micro-F1 Score", fontsize=14, fontweight="bold", color="#333")
ax1.set_xlabel("Model", fontsize=14, fontweight="bold", color="#333")
ax1.set_title("Average Micro-F1 Score", fontsize=16, fontweight="bold", color="#222")
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", linestyle="--", alpha=0.4, color="#aaa")  
ax1.tick_params(axis="x", labelsize=12, labelcolor="#444")
ax1.tick_params(axis="y", labelsize=12, labelcolor="#444")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Macro-F1 Score
bars2 = ax2.bar(results_df["model"], results_df["avg_test_macro_f1"], 
                color=[pastel_colors.get(model, "#D6E8FF") for model in results_df["model"]],
                linewidth=1.2, alpha=0.85)


for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", 
             ha="center", va="bottom", fontsize=12, fontweight="bold", color="#444")

ax2.set_ylabel("Average Macro-F1 Score", fontsize=14, fontweight="bold", color="#333")
ax2.set_xlabel("Model", fontsize=14, fontweight="bold", color="#333")
ax2.set_title("Average Macro-F1 Score", fontsize=16, fontweight="bold", color="#222")
ax2.set_ylim(0, 1.0)
ax2.grid(axis="y", linestyle="--", alpha=0.4, color="#aaa")  
ax2.tick_params(axis="x", labelsize=12, labelcolor="#444")
ax2.tick_params(axis="y", labelsize=12, labelcolor="#444")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "f1_3_models.png"), dpi=300)
plt.show()


########### EVALUATION ####################

### Step 1: Generate Predictions

with open(os.path.join(DATA_DIR, "Annotations/Dev/json_format/dev.json"), "r")as f:
    dev_data = json.load(f)

PRED_DIR = os.path.join(RESULTS_DIR,"predictions/")
os.makedirs(PRED_DIR, exist_ok=True)

### Get final metrics for all models and all seeds and then averaged over seed (but also max values) ###################
for model_name in MODEL_CONFIGS.keys():
    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str) 
    model_name_full = MODEL_CONFIGS[model_name]["model_name"]
    for seed in seeds:
        model_path = f"{MODEL_SAVE_DIR}/{model_name}_{seed}.pt" # get saved best model state for that random seed from MODEL_SAVE_DIR
        model = AutoModelForTokenClassification.from_pretrained(model_name_full, num_labels=len(label_map))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() 

        predictions_output = {}
        
        for article_id, article_data in dev_data.items():
            metadata = article_data.get("metadata", {})
            article_entities = []
            
            if "title" in metadata and metadata["title"]:
                title_text = metadata["title"] 
                title_entities = predict_entities(title_text, "title", model, tokenizer, DEVICE, label_map) 
                article_entities.extend(title_entities)
            
            if "abstract" in metadata and metadata["abstract"]:
                abstract_text = metadata["abstract"]
                abstract_entities = predict_entities(abstract_text, "abstract", model, tokenizer, DEVICE, label_map)
                article_entities.extend(abstract_entities)
            
            predictions_output[article_id] = {"entities": article_entities}
        
        path =os.path.join(PRED_DIR, f"Predictions_{model_name}_{seed}.json")
        with open(path, "w") as f:
            json.dump(predictions_output, f, indent=4)

### Step 2: calculate Metrics with the gutbrainie2025 script (modified after https://github.com/MMartinelli-hub/GutBrainIE_2025_Baseline/blob/main/Eval/evaluate.py)

PREDICTION_PATH_DIR = PRED_DIR

with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as file:
    ground_truth = json.load(file)

round_to_decimal_position = 4

bert_model_results = {
    "BERT": [],
    "BioBERT": [],
    "PubMedBERT": []
}

# To save the best scores from the predictions
best_bert_scores = {
    "BERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0},
    "BioBERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0},
    "PubMedBERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0}
}

for filename in os.listdir(PREDICTION_PATH_DIR):
    file_path = os.path.join(PREDICTION_PATH_DIR, filename)
    
    if filename.endswith(".json"):
        print(f"\n Evaluating: {filename}") # print the metrics for each model and seed 
        model_name, seed = filename.split('_')[1], filename.split('_')[1].split('.')[0]
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_1_NER(file_path)
        bert_model_results[model_name].append([precision, recall, f1, micro_precision, micro_recall, micro_f1])

        best_bert_scores[model_name]["precision"] = max(best_bert_scores[model_name]["precision"], precision)
        best_bert_scores[model_name]["recall"] = max(best_bert_scores[model_name]["recall"], recall)
        best_bert_scores[model_name]["f1"] = max(best_bert_scores[model_name]["f1"], f1)
        best_bert_scores[model_name]["micro_precision"] = max(best_bert_scores[model_name]["micro_precision"], micro_precision)
        best_bert_scores[model_name]["micro_recall"] = max(best_bert_scores[model_name]["micro_recall"], micro_recall)
        best_bert_scores[model_name]["micro_f1"] = max(best_bert_scores[model_name]["micro_f1"], micro_f1)

        print("\n\n=== 6_1_NER ===")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

best_bert_scores_path = os.path.join(RESULTS_DIR,"best_bert_scores.json")
all_bert_scores_path = os.path.join(RESULTS_DIR,"all_bert_scores.json")

with open(best_bert_scores_path, "w") as json_file:
    json.dump(best_bert_scores, json_file, indent=4) # the best results

with open(all_bert_scores_path, "w") as json_file:
    json.dump(bert_model_results, json_file, indent=4) # all results per seed


# Step 3: get averages per model across seeds and plot them next to the results of the other models (results for dev set)

BiLSTM_results = [0.6399, 0.514, 0.5262, 0.6914, 0.6598, 0.6752]
GLiNER_results = [0.6627, 0.7473, 0.6917, 0.7561, 0.8272, 0.7901] # cf. https://hereditary.dei.unipd.it/challenges/gutbrainie/2025/, assessed Apr 11, 2025
BERT_avg = calculate_averages("BERT", bert_model_results)
BioBERT_avg = calculate_averages("BioBERT", bert_model_results)
PubMedBERT_avg = calculate_averages("PubMedBERT", bert_model_results)


# Step 4: Plot all metrics for all models

model_names = ["BiLSTM-CRF", "GLiNER", "BERT", "BioBERT", "PubMedBERT"]
metrics = ["Macro-precision", "Macro-recall", "Macro-F1", "Micro-precision", "Micro-recall", "Micro-F1"]

results = [
    BiLSTM_results,
    GLiNER_results,
    BERT_avg,
    BioBERT_avg,
    PubMedBERT_avg
]

pastel_colors = { 
    "BERT": "#D6E8FF",     
    "BioBERT": "#E4D5FF",  
    "PubMedBERT": "#FFCCE5" 
}

fixed_colors = {
    "BiLSTM-CRF": "#FFCC80",
    "GLiNER": "#FFEB99",
}


macro_metrics = metrics[:3]
micro_metrics = metrics[3:]
macro_index = np.arange(len(macro_metrics))
micro_index = np.arange(len(micro_metrics))

bar_width = 0.18


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))


# 1) plot macro scores in the first plot 

for i, (model_name, result) in enumerate(zip(model_names, results)):
    color = fixed_colors.get(model_name, pastel_colors.get(model_name, "#D6E8FF"))
    macro_values = result[:3]

    bars = ax1.bar(macro_index + i * bar_width, macro_values, bar_width, label=model_name, color=color, edgecolor="lightgrey")
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax1.set_ylabel("Scores", fontsize=14)
ax1.set_title("Macro Scores", fontsize=16, fontweight="bold")
ax1.set_xticks(macro_index + 2 * bar_width)
ax1.set_xticklabels(macro_metrics, fontsize=12, fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.legend(title="Models")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# 2) plot micro scores below
for i, (model_name, result) in enumerate(zip(model_names, results)):
    color = fixed_colors.get(model_name, pastel_colors.get(model_name, "#D6E8FF"))
    micro_values = result[3:]

    bars = ax2.bar(micro_index + i * bar_width, micro_values, bar_width, color=color, edgecolor="lightgrey")

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax2.set_ylabel("Scores", fontsize=14)
ax2.set_title("Micro Scores", fontsize=16, fontweight="bold")
ax2.set_xticks(micro_index + 2 * bar_width)
ax2.set_xticklabels(micro_metrics, fontsize=12, fontweight="bold")
ax2.set_ylim(0, 1.0)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.subplots_adjust(hspace=0.4) # bit more space between subplots
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"model_comparison_plot.png"))
plt.show()



