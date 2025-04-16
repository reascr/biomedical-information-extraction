from config import MODEL_CONFIGS, label_map, seeds
from datasets import create_dataloaders_RE, get_adjusted_indices, mark_entities
from scripts.utils import set_seed
from utils import plot_train_val_metrics, train_and_evaluate_RE, calculate_averages
from get_predictions_RE import get_entities, generate_candidate_pairs
from models import RelationClassifier
from evaluation_gutbrainie2025 import eval_submission_6_2_binary_tag_RE, GROUND_TRUTH_PATH
from dotenv import load_dotenv
import os
import json
import torch
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from transformers import AutoTokenizer, AutoModel


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

RESULTS_DIR = os.path.join(script_dir,"..", "results", "RE", "BERT-models") 
os.makedirs(RESULTS_DIR, exist_ok=True)
OPTUNA_RESULTS_DIR = os.path.join(script_dir, "..", "results", "RE", "optuna")
print(f"File path: {os.path.abspath(OPTUNA_RESULTS_DIR)}")
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")

NER_PREDICTIONS_DIR = os.path.join(script_dir,"..", "results", "NER", "BERT-models", "predictions")
RE_PREDICTIONS_DIR = os.path.join(script_dir,"..", "results", "RE", "BERT-models", "predictions") 
os.makedirs(RE_PREDICTIONS_DIR, exist_ok=True)

load_dotenv()  # load the .env file with the wandb key
wandb.login(key=os.getenv("WANDB_API_KEY"))

# directory of the NER models we can use in the pipeline
MODEL_SAVE_DIR = os.path.join("..", "results", "NER", "best-models") 


############ TRAINING #######################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
NUM_EPOCHS = 6
THRESHOLD = 0.5 # threshold for predicting positive class
special_tokens = ['<ent1>', '</ent1>', '<ent2>', '</ent2>'] # entity markers that will be added to the tokenizers


best_models = {}

results = []

for model_name in MODEL_CONFIGS.keys():
    file_path = os.path.join(OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        
    best_params = data[model_name]["best_params"] # get the best paramaters from the optuna study

    BATCH_SIZE = best_params["BATCH_SIZE"]
    LR = best_params["LR"]
    WEIGHT_DECAY = best_params["weight_decay"]
    DROPOUT = best_params["dropout"]
    MAX_NORM = best_params["max_norm"]

    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    tokenizer.add_tokens(special_tokens)
    tokenizer_voc_size = len(tokenizer)
    vocab = tokenizer.get_vocab()

    last_four_token_ids = sorted(vocab.values())[-4:] # extract the last four token ids (=entity markers)
    ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id = last_four_token_ids

    #train_dataloader, val_dataloader, test_dataloader = create_dataloaders_RE(BATCH_SIZE,tokenizer, DEVICE) # create the data loaders

    test_micro_f1_scores = []
    test_macro_f1_scores = []
    all_train_losses = []
    all_val_losses = []
    all_train_f1s_micro = []
    all_train_f1s_macro = []
    all_val_f1s_micro = []
    all_val_f1s_macro = []
    
    best_micro_f1 = -1
    best_macro_f1 = -1
    best_model_state = None
    best_model_seed = None

    for seed in seeds:
        set_seed(seed)
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders_RE(BATCH_SIZE,tokenizer, DEVICE) # create the data loaders
        wandb.init(
        project="Relation_Classification",
        entity="lp2",
        config={
        "model": model_name,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "max_norm": MAX_NORM
        },
        name=f"{model_name}_seed{seed}_{THRESHOLD}",
        group=model_name,  # Groups all runs for a given model together
        tags=[model_name, f"seed-{seed}"])
        config = wandb.config

        # train and evaluate model
        model, test_micro_f1, test_macro_f1, train_losses, val_losses, train_f1s_micro, train_f1_macro, val_f1s_micro, val_f1_macro = train_and_evaluate_RE(
            model_name, tokenizer_voc_size, seed, train_dataloader, val_dataloader, test_dataloader, LR, WEIGHT_DECAY, NUM_EPOCHS, DROPOUT, DEVICE, MAX_NORM, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, THRESHOLD) 
        
        test_micro_f1_scores.append(test_micro_f1)
        test_macro_f1_scores.append(test_macro_f1)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_f1s_micro.append(train_f1s_micro)
        all_train_f1s_macro.append(train_f1_macro) 
        all_val_f1s_micro.append(val_f1s_micro) 
        all_val_f1s_macro.append(val_f1_macro)
        
        plot_train_val_metrics(train_losses, val_losses, train_f1s_micro, val_f1s_micro, model_name, seed)

        # save the best model 
        if test_micro_f1 > best_micro_f1:
            best_micro_f1 = test_micro_f1
            best_model_state = model.state_dict()
            best_model_seed = seed

        #seed_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{seed}.pt") 
        #torch.save(model.state_dict(), seed_model_path) # saves all models 

    best_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.pt")
    best_models[model_name] = best_model_path
    torch.save(best_model_state, best_model_path) # Only save the best model of all seeds for error analysis

    # get mean and std of micro and macro F1s
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


wandb.finish()

results_df = pd.DataFrame(results)
results_json_path = "/kaggle/working/relation_classification_results.json"
results_df.to_json(results_json_path, orient="records", indent=4)


########### EVALUATION #####################
THRESHOLD_INFERENCE = 0.5 # maybe for first run, we keep it at 0.5/0.5 to get comparative values :) 
use_ground_truth = False # uses ground truth NER annotations in case ground truth is set to True, otherwise uses NER predictions

if use_ground_truth:
    with open(os.path.join(DATA_DIR, "Annotations/Dev/json_format/dev.json"), "r")as f:
        ground_truth_data = json.load(f)

PRED_DIR = os.path.join(RESULTS_DIR,"predictions/")
os.makedirs(PRED_DIR, exist_ok=True)

### Get final metrics for all models and all seeds and then averaged over seed (but also max values) ###################
for model_name in MODEL_CONFIGS.keys():
    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str) 
    model_name_full = MODEL_CONFIGS[model_name]["model_name"]

    tokenizer_voc_size = len(tokenizer)
    vocab = tokenizer.get_vocab()

    last_four_token_ids = sorted(vocab.values())[-4:] # extract the last four token ids (=entity markers)
    ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id = last_four_token_ids

    file_path = os.path.join(OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")

    with open(file_path, "r") as f:
        data = json.load(f)
        
    best_params = data[model_name]["best_params"]
    DROPOUT = best_params["dropout"] # get the best dropout for the model intitalization below

    for seed in seeds:
        model_path = f"{MODEL_SAVE_DIR}/{model_name}_{seed}.pt" # get saved best model state for that random seed from MODEL_SAVE_DIR
        
        model = AutoModel.from_pretrained(model_name_full)
        model.resize_token_embeddings(tokenizer_voc_size)  # adjust embeddings of the model for special tokens

        hidden_size = model.config.hidden_size
    
        model = RelationClassifier(model, hidden_size, DROPOUT, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id).to(DEVICE)
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() 

        predictions = {}

        if not use_ground_truth:
            ner_prediction_file = os.path.join(NER_PREDICTIONS_DIR, f"Predictions_{model_name_full}_{seed}")
            with open(ner_prediction_file, "r") as g:
                ner_predictions = json.load(g)

        for abstract_id, article_data in tqdm(ground_truth_data.items(), desc="Processing Abstracts", unit="abstract"):
            if use_ground_truth:
                entities = get_entities(abstract_id, article_data)
            else:
                entities = get_entities(abstract_id, ner_predictions)

            metadata = article_data.get("metadata", {})
            title = metadata.get("title", "")
            abstract = metadata.get("abstract", "")
            full_text = (title + " " + abstract).strip()  # get the combination of text and abstract
            offset = len(title) + 1 # offset for abstract positions

            candidate_pairs = generate_candidate_pairs(entities)
                
            for entity1, entity2 in candidate_pairs:
                subj_start_idx, subj_end_idx = get_adjusted_indices(entity1, offset)
                obj_start_idx, obj_end_idx  = get_adjusted_indices(entity2, offset)
                        
                # insert entity markers in the text for entity1 and entity2
                marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    
                inputs = tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                # run through the relation classifier
                with torch.no_grad():
                    output = model(inputs["input_ids"], inputs["attention_mask"])
                relation_exists = (torch.sigmoid(output) > THRESHOLD_INFERENCE).item() # returns 1 if relation, 0 else
                    
                if relation_exists:
                    rel_info = {"subject_label": entity1["label"], "object_label": entity2["label"]}
                    if abstract_id not in predictions:
                        predictions[abstract_id] = {"binary_tag_based_relations": []}
                    if rel_info not in predictions[abstract_id]["binary_tag_based_relations"]:
                        predictions[abstract_id]["binary_tag_based_relations"].append(rel_info)

        pred_filename = os.path.join(PRED_DIR, f"Predictions_{model_name}_{seed}.json")
        with open(pred_filename, "w") as f:
            json.dump(predictions, f, indent=4)


###### Step 2: calculate Metrics with the gutbrainie2025 script (modified after https://github.com/MMartinelli-hub/GutBrainIE_2025_Baseline/blob/main/Eval/evaluate.py)

PREDICTION_PATH_DIR = RE_PREDICTIONS_DIR

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
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_2_binary_tag_RE(file_path)
        bert_model_results[model_name].append([precision, recall, f1, micro_precision, micro_recall, micro_f1])

        best_bert_scores[model_name]["precision"] = max(best_bert_scores[model_name]["precision"], precision)
        best_bert_scores[model_name]["recall"] = max(best_bert_scores[model_name]["recall"], recall)
        best_bert_scores[model_name]["f1"] = max(best_bert_scores[model_name]["f1"], f1)
        best_bert_scores[model_name]["micro_precision"] = max(best_bert_scores[model_name]["micro_precision"], micro_precision)
        best_bert_scores[model_name]["micro_recall"] = max(best_bert_scores[model_name]["micro_recall"], micro_recall)
        best_bert_scores[model_name]["micro_f1"] = max(best_bert_scores[model_name]["micro_f1"], micro_f1)

        print("\n\n=== 6_2_binary_tag_RE ===\n")
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


# Plot all metrics when I trained the baseline...



