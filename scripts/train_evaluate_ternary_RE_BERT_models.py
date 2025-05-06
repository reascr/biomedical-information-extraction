from config import MODEL_CONFIGS, label_map, seeds
from datasets import create_dataloaders_RE_ternary, get_adjusted_indices, mark_entities
from models import RelationClassifier_CLS_ent1_ent2_avg_pooled, RelationClassifier_CLSOnly, RelationClassifier_ent1_ent2_start_token, RelationClassifier_ent1_ent2_average_pooled, RelationClassifier_ternary_ent1_ent2_average_pooled
from utils import set_seed
from utils import plot_train_val_metrics, train_and_evaluate_RE_ternary
from get_predictions_RE import get_entities, generate_candidate_pairs
from evaluation_gutbrainie2025 import eval_submission_6_3_ternary_tag_RE, eval_submission_6_4_ternary_mention_RE, GROUND_TRUTH_PATH
from dotenv import load_dotenv
import os
import json
import torch
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
INDEX_TO_LABEL = None # stores the index to predicate mapping

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

RESULTS_DIR = os.path.join(script_dir,"..", "results", "RE-ternary", "BERT-models") 
os.makedirs(RESULTS_DIR, exist_ok=True)
OPTUNA_RESULTS_DIR = os.path.join(script_dir, "..", "results", "RE-ternary", "optuna")
print(f"File path: {os.path.abspath(OPTUNA_RESULTS_DIR)}")
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")

NER_PREDICTIONS_DIR = os.path.join(script_dir,"..", "results", "NER", "BERT-models", "predictions")
RE_PREDICTIONS_DIR = os.path.join(script_dir,"..", "results", "RE-ternary", "BERT-models", "predictions") 
os.makedirs(RE_PREDICTIONS_DIR, exist_ok=True)

# directory of the NER models we can use in the pipeline
MODEL_SAVE_DIR = os.path.join("..", "results", "NER", "best-models") 


# folders for ucloud drive
DRIVE_MODEL_SAVE_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "models")) # save on ucloud drive
os.makedirs(DRIVE_MODEL_SAVE_DIR, exist_ok=True)
DRIVE_NER_PREDICTIONS_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "NER", "results", "predictions")) # save on ucloud drive
DRIVE_RE_PREDICTIONS_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "results", "predictions")) # save on ucloud drive
os.makedirs(DRIVE_RE_PREDICTIONS_DIR, exist_ok=True)
#DRIVE_OPTUNA_RESULTS_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "optuna"))
DRIVE_OPTUNA_RESULTS_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "optuna"))
DRIVE_RESULTS_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "results", "BERT-models")) 
os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)

DRIVE_PRED_DIR_MENTION_BASED = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "results", "BERT-models", "mention_based_preds")) 
os.makedirs(DRIVE_PRED_DIR_MENTION_BASED, exist_ok=True)

DRIVE_PRED_DIR_MENTION_BASED_EXTENDED = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "results", "BERT-models", "mention_based_preds_extended")) 
os.makedirs(DRIVE_PRED_DIR_MENTION_BASED_EXTENDED, exist_ok=True)

load_dotenv()  # load the .env file with the wandb key
wandb.login(key=os.getenv("WANDB_API_KEY"))



############ TRAINING #######################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
NUM_EPOCHS = 6
NUM_LABELS = None 
special_tokens = ['<ent1>', '</ent1>', '<ent2>', '</ent2>'] # entity markers that will be added to the tokenizers


best_models = {}

results = []

for model_name in MODEL_CONFIGS.keys():
    file_path = os.path.join(DRIVE_OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        
    # get best parameters from the optuna study
    best_params = data["best_params"]
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

    ent1_start_id = tokenizer.convert_tokens_to_ids("<ent1>")
    ent1_end_id   = tokenizer.convert_tokens_to_ids("</ent1>")
    ent2_start_id = tokenizer.convert_tokens_to_ids("<ent2>")
    ent2_end_id   = tokenizer.convert_tokens_to_ids("</ent2>")

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
        train_dataloader, val_dataloader, test_dataloader, num_labels, index_to_label = create_dataloaders_RE_ternary(BATCH_SIZE,tokenizer, DEVICE) # create the data loaders
        INDEX_TO_LABEL = index_to_label
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
        name=f"{model_name}_seed{seed}_ternary",
        group=model_name,  # Groups all runs for a given model together
        tags=[model_name, f"seed-{seed}"])
        config = wandb.config
        NUM_LABELS = num_labels

        # train and evaluate model
        model, test_micro_f1, test_macro_f1, train_losses, val_losses, train_f1s_micro, train_f1_macro, val_f1s_micro, val_f1_macro = train_and_evaluate_RE_ternary(
            model_name, tokenizer_voc_size, seed, train_dataloader, val_dataloader, test_dataloader, LR, WEIGHT_DECAY, NUM_EPOCHS, DROPOUT, DEVICE, MAX_NORM, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels, index_to_label) 
        wandb.finish()
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
        if test_macro_f1 > best_macro_f1:
            best_macro_f1 = test_macro_f1

        seed_model_path = os.path.join(DRIVE_MODEL_SAVE_DIR, f"{model_name}_{seed}.pt") 
        torch.save(model.state_dict(), seed_model_path) # saves all models 

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
        "all test f1s": test_micro_f1_scores,
        "all test macro f1s": test_macro_f1_scores
    })


results_df = pd.DataFrame(results)
results_json_path = os.path.join(DRIVE_RESULTS_DIR, "RE_training_results.json")
results_df.to_json(results_json_path, orient="records", indent=4)


########### EVALUATION #####################


use_ground_truth = False # uses ground truth NER annotations in case ground truth is set to True, otherwise uses NER predictions

with open(os.path.join(DATA_DIR, "Annotations/Dev/json_format/dev.json"), "r")as f:
    ground_truth_data = json.load(f)

PRED_DIR = os.path.join(DRIVE_RESULTS_DIR,"predictions/")
os.makedirs(PRED_DIR, exist_ok=True)


### Get final metrics for all models and all seeds and then averaged over seed (but also max values) ###################
for model_name in MODEL_CONFIGS.keys():
    tokenizer_str = MODEL_CONFIGS[model_name]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    tokenizer.add_tokens(special_tokens) 
    tokenizer_voc_size = len(tokenizer)
    model_name_full = MODEL_CONFIGS[model_name]["model_name"]

    ent1_start_id = tokenizer.convert_tokens_to_ids("<ent1>")
    ent1_end_id   = tokenizer.convert_tokens_to_ids("</ent1>")
    ent2_start_id = tokenizer.convert_tokens_to_ids("<ent2>")
    ent2_end_id   = tokenizer.convert_tokens_to_ids("</ent2>")

    file_path = os.path.join(DRIVE_OPTUNA_RESULTS_DIR, f"optuna_results_{model_name}.json")

    with open(file_path, "r") as f:
        data = json.load(f)
        
    best_params = data["best_params"]
    DROPOUT = best_params["dropout"] # get the best dropout for the model intitalization below

    for seed in seeds:
        model_path = f"{DRIVE_MODEL_SAVE_DIR}/{model_name}_{seed}.pt" # get saved best model state for that random seed from MODEL_SAVE_DIR
        
        model = AutoModel.from_pretrained(model_name_full)
        model.resize_token_embeddings(tokenizer_voc_size)  # adjust embeddings of the model for special tokens

        hidden_size = model.config.hidden_size
 
        model = RelationClassifier_ternary_ent1_ent2_average_pooled(model, hidden_size, DROPOUT, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels).to(DEVICE) # change RelationClassifier type here NUM_LABELS
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() 

        predictions = {} # predictions for ternary tag based RE (task 6.3) to match evaluation format of the challenge
        predictions_mention_based = {} # predictions for ternary mention based RE (task 6.4) to match evaluation format of the challenge
        predictions_mention_based_extended = {} # predictions with score and start and end indices of entities

        if not use_ground_truth:
            ner_prediction_file = os.path.join(DRIVE_NER_PREDICTIONS_DIR, f"Predictions_{model_name}_{seed}.json")
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
                probabilities = torch.softmax(output, dim=1) #  get probability distribution from softmax
                maximum_pred_index = torch.argmax(probabilities, dim=1).item() # gets the max index of the maximum probability, item() converts to scalar
                predicate = INDEX_TO_LABEL[maximum_pred_index]
                score = probabilities[0, maximum_pred_index].item()

                if predicate != "no relation":
                    rel_info = {"subject_label": entity1["label"], "predicate":predicate ,"object_label": entity2["label"]}
                    rel_info_mention_based = {
                        "subject_text_span": entity1["text_span"],
                        "subject_label": entity1["label"],
                        "predicate":     predicate,
                        "object_text_span":  entity2["text_span"],
                        "object_label":  entity2["label"]
                    }
                    rel_info_mention_based_extended = {
                        "subject_text_span": entity1["text_span"],
                        "subject_label": entity1["label"],
                        "predicate":     predicate,
                        "object_text_span":  entity2["text_span"],
                        "object_label":  entity2["label"],
                        "subject_start_index": entity1["start_idx"],
                        "subject_end_index":   entity1["end_idx"],
                        "object_start_index":  entity2["start_idx"],
                        "object_end_index":    entity2["end_idx"],
                        "score": score
                    }

                    if abstract_id not in predictions:
                        predictions[abstract_id] = {"ternary_tag_based_relations": []}
                        predictions_mention_based[abstract_id] = {"ternary_mention_based_relations": []}
                        predictions_mention_based_extended[abstract_id] = {"ternary_mention_based_relations_extended": []}
                    
                    predictions_mention_based_extended[abstract_id]["ternary_mention_based_relations_extended"].append(rel_info_mention_based_extended)

                    if rel_info_mention_based not in predictions_mention_based[abstract_id]["ternary_mention_based_relations"]:
                        predictions_mention_based[abstract_id]["ternary_mention_based_relations"].append(rel_info_mention_based)

                    if rel_info not in predictions[abstract_id]["ternary_tag_based_relations"]:
                        predictions[abstract_id]["ternary_tag_based_relations"].append(rel_info) # here we only want a set of predictions because it is tag based

        pred_filename = os.path.join(PRED_DIR, f"Predictions_ternary_RE_{model_name}_{seed}.json")
        pred_filename_mention_based = os.path.join(DRIVE_PRED_DIR_MENTION_BASED, f"Predictions_ternary_RE_mention_based_{model_name}_{seed}.json")
        pred_filename_mention_based_extended = os.path.join(DRIVE_PRED_DIR_MENTION_BASED_EXTENDED, f"Predictions_ternary_RE_mention_based_extended_{model_name}_{seed}.json")
        with open(pred_filename, "w") as f:
            json.dump(predictions, f, indent=4)
        with open(pred_filename_mention_based, "w") as f:
            json.dump(predictions_mention_based, f, indent=4)
        with open(pred_filename_mention_based_extended, "w") as f:
            json.dump(predictions_mention_based_extended, f, indent=4)


###### Step 2: calculate Metrics with the gutbrainie2025 script (modified after https://github.com/MMartinelli-hub/GutBrainIE_2025_Baseline/blob/main/Eval/evaluate.py)

### 1. ternary tag based RE

PREDICTION_PATH_DIR = PRED_DIR

with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as file:
    ground_truth = json.load(file)

round_to_decimal_position = 4

bert_model_results = {
    "BERT": [],
    "BioBERT": [],
    "PubMedBERT": []
}

# save the best scores from the predictions
best_bert_scores = {
    "BERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0},
    "BioBERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0},
    "PubMedBERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0}
}


for filename in os.listdir(PREDICTION_PATH_DIR):
    file_path = os.path.join(PREDICTION_PATH_DIR, filename)
    
    if filename.endswith(".json"):
        parts = filename.split('_')
        model_name = parts[3]
        seed = parts[4].split('.')[0]
        print(f"\n Evaluating: {filename}") # print the metrics for each model and seed 
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_3_ternary_tag_RE(file_path)
        bert_model_results[model_name].append([precision, recall, f1, micro_precision, micro_recall, micro_f1])

        best_bert_scores[model_name]["precision"] = max(best_bert_scores[model_name]["precision"], precision)
        best_bert_scores[model_name]["recall"] = max(best_bert_scores[model_name]["recall"], recall)
        best_bert_scores[model_name]["f1"] = max(best_bert_scores[model_name]["f1"], f1)
        best_bert_scores[model_name]["micro_precision"] = max(best_bert_scores[model_name]["micro_precision"], micro_precision)
        best_bert_scores[model_name]["micro_recall"] = max(best_bert_scores[model_name]["micro_recall"], micro_recall)
        best_bert_scores[model_name]["micro_f1"] = max(best_bert_scores[model_name]["micro_f1"], micro_f1)

        print("\n\n=== 6_3_ternary_tag_RE ===\n")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

best_bert_scores_path = os.path.join(DRIVE_RESULTS_DIR,"best_bert_scores_ternary_tag_based_RE.json")
all_bert_scores_path = os.path.join(DRIVE_RESULTS_DIR,"all_bert_scores__ternary_tag_based_RE.json")

with open(best_bert_scores_path, "w") as json_file:
    json.dump(best_bert_scores, json_file, indent=4) # the best results

with open(all_bert_scores_path, "w") as json_file:
    json.dump(bert_model_results, json_file, indent=4) # all results per seed


### 2. ternary mention based RE

PREDICTION_PATH_DIR = DRIVE_PRED_DIR_MENTION_BASED # use the mention based predictions for ternary RE

with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as file:
    ground_truth = json.load(file)

round_to_decimal_position = 4

bert_model_results = {
    "BERT": [],
    "BioBERT": [],
    "PubMedBERT": []
}

# save the best scores from the predictions
best_bert_scores = {
    "BERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0},
    "BioBERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0},
    "PubMedBERT": {"precision": 0, "recall": 0, "f1": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0}
}

for filename in os.listdir(PREDICTION_PATH_DIR):
    file_path = os.path.join(PREDICTION_PATH_DIR, filename)
    
    if filename.endswith(".json"):
        parts = filename.split('_')
        model_name = parts[5]
        seed = parts[6].split('.')[0]
        print(f"\n Evaluating: {filename}") # print the metrics for each model and seed 
        precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_4_ternary_mention_RE(file_path)
        bert_model_results[model_name].append([precision, recall, f1, micro_precision, micro_recall, micro_f1])

        best_bert_scores[model_name]["precision"] = max(best_bert_scores[model_name]["precision"], precision)
        best_bert_scores[model_name]["recall"] = max(best_bert_scores[model_name]["recall"], recall)
        best_bert_scores[model_name]["f1"] = max(best_bert_scores[model_name]["f1"], f1)
        best_bert_scores[model_name]["micro_precision"] = max(best_bert_scores[model_name]["micro_precision"], micro_precision)
        best_bert_scores[model_name]["micro_recall"] = max(best_bert_scores[model_name]["micro_recall"], micro_recall)
        best_bert_scores[model_name]["micro_f1"] = max(best_bert_scores[model_name]["micro_f1"], micro_f1)

        print("\n\n=== 6_4_ternary_mention_based_RE ===\n")
        print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
        print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
        print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
        print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
        print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
        print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")

best_bert_scores_path = os.path.join(DRIVE_RESULTS_DIR,"best_bert_scores_ternary_mention_based_RE.json")
all_bert_scores_path = os.path.join(DRIVE_RESULTS_DIR,"all_bert_scores__ternary_tag_mention_based_RE.json")

with open(best_bert_scores_path, "w") as json_file:
    json.dump(best_bert_scores, json_file, indent=4) # the best results

with open(all_bert_scores_path, "w") as json_file:
    json.dump(bert_model_results, json_file, indent=4) # all results per seed





