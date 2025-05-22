from config import MODEL_CONFIGS, label_map, label_map_RE
from models import RelationClassifier_CLSOnly, RelationClassifier_CLS_ent1_ent2_avg_pooled, RelationClassifier_ent1_ent2_average_pooled, RelationClassifier_ent1_ent2_start_token, RelationClassifier_ternary_ent1_ent2_average_pooled
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import pearsonr
from transformers import BertForTokenClassification, AutoModel, get_scheduler, AdamW


script_dir = os.path.dirname(os.path.abspath(__file__))
DRIVE_PLOT_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "NER", "plots")) # save on ucloud drive
DRIVE_PLOT_DIR_RE = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE-ternary", "plots"))
os.makedirs(DRIVE_PLOT_DIR_RE, exist_ok=True)


DRIVE_RESULTS_DIR_NER = os.path.join("..", "results", "NER", "BERT-models") # specify your result directory. All plots will be saved here
os.makedirs(DRIVE_RESULTS_DIR_NER, exist_ok=True)

DRIVE_RESULTS_DIR_RE = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "work", "RE", "results", "BERT-models")) 
os.makedirs(DRIVE_RESULTS_DIR_RE, exist_ok=True)


# cf. https://github.com/heraclex12/R-BERT-Relation-Classification/blob/master/BERT_for_Relation_Classification.ipynb, assessed April 10, 2025
def set_seed(seed):
    """Sets a random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# modified after Chollet 2018: 75
def plot_train_val_metrics(train_losses, val_losses, train_f1s, val_f1s, model_name, seed, plot_save_path=None):
    """Plots the training and validation micro F1 and loss."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    epochs = range(1, len(train_losses) + 1)

    axs[0].plot(epochs,train_losses, label=f'Training loss', color="cyan")
    axs[0].plot(epochs,val_losses, label=f'Validation loss', linestyle='--', color="magenta")
    #axs[0].set_title(f'{model_name} - Train and Val Loss micro (Seed {seed})')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_xticks(epochs)
    #axs[0].set_xlim(1, len(train_losses))

    axs[1].plot(epochs,train_f1s, label=f'Training Micro F1', color="cyan")
    axs[1].plot(epochs,val_f1s, label=f'Validation Micro F1', linestyle='--', color="magenta")
    #axs[1].set_title(f'{model_name} - Train and Val F1 micro Score (Seed {seed})')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Micro F1")
    axs[1].legend()
    axs[1].set_xticks(epochs)
    #axs[1].set_xlim(1, len(train_f1s))

    plt.tight_layout()

    filename = f"train_val_loss_f1_{model_name}_{seed}.png"
    if plot_save_path:
        save_path = os.path.join(plot_save_path, filename)
    else:
        save_path = os.path.join(DRIVE_RESULTS_DIR_RE, filename)
    plt.savefig(save_path)
    plt.show()


def calculate_averages(model_name, bert_model_results):
    """Calculates averages for the three BERT models for each metric"""
    all_results = bert_model_results[model_name]
    if len(all_results) == 0:
        return [0, 0, 0, 0, 0, 0]
    
    avg_precision = sum([result[0] for result in all_results]) / len(all_results)
    avg_recall = sum([result[1] for result in all_results]) / len(all_results)
    avg_f1 = sum([result[2] for result in all_results]) / len(all_results)
    avg_micro_precision = sum([result[3] for result in all_results]) / len(all_results)
    avg_micro_recall = sum([result[4] for result in all_results]) / len(all_results)
    avg_micro_f1 = sum([result[5] for result in all_results]) / len(all_results)
    
    return [avg_precision, avg_recall, avg_f1, avg_micro_precision, avg_micro_recall, avg_micro_f1]

def compute_f1(labels, preds):
    labels = np.array(labels).flatten()
    preds = np.array(preds).flatten()
    
    # remove padding tokens (-100)
    valid_indices = labels != -100
    labels = labels[valid_indices]
    preds = preds[valid_indices]

    return f1_score(labels, preds, average="micro"), f1_score(labels, preds, average="macro")


def compute_per_class_f1(labels, preds, label_map):
    labels = np.array(labels).flatten()
    preds = np.array(preds).flatten()
    valid_indices = labels != -100 # remove padding tokens (-100)
    labels = labels[valid_indices]
    preds = preds[valid_indices]
    
    # get precision, recall, f1 and frequency (=support) per class
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None) # this makes sure that we calculate per-label precisions, recalls, F1-scores and supports (=frequencies)
    
    # map the label indexes to class names for better reading
    per_class_scores = {}
    for idx, label_name in label_map.items():
        per_class_scores[label_name] = {
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx],
            "support": support[idx]
        }
    return per_class_scores
    

############### NER Training #################

def train_and_evaluate_NER(model_name, seed, train_dataloader, val_dataloader, test_dataloader, lr, weight_decay, num_epochs, dropout, num_labels, device, max_norm = 1.0, track_wandb=True):
    #set_seed(seed)
    model = BertForTokenClassification.from_pretrained(MODEL_CONFIGS[model_name]["model_name"], num_labels=num_labels, ignore_mismatched_sizes=True)
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model.to(device)

    best_model_state = None # track best model state in case of overfitting

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr schedule with warm up in the first 10% steps (cf. Gu et al. 2021)
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # early stopping
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_labels = []
        total_train_preds = []

        # cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #print(outputs[0]) # sanity check: verify whether loss is good 3.2689 -ln(1/27)
            logits = outputs.logits
            #print(logits.shape)# sanity check: (batch_size, sequence_length, num_labels) 

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_labels.extend(labels.cpu().numpy())
            total_train_preds.extend(torch.argmax(logits, dim=2).cpu().numpy())

        train_losses.append(total_train_loss / len(train_dataloader))
        train_micro_f1, train_macro_f1 = compute_f1(total_train_labels, total_train_preds)
        train_f1s.append(train_micro_f1)

        # validation
        model.eval()
        total_val_labels = []
        total_val_preds = []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                total_val_loss += loss.item()
                total_val_labels.extend(labels.cpu().numpy())
                total_val_preds.extend(torch.argmax(logits, dim=2).cpu().numpy())

        val_losses.append(total_val_loss / len(val_dataloader))
        val_micro_f1, val_macro_f1 = compute_f1(total_val_labels, total_val_preds)
        val_f1s.append(val_micro_f1)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_losses[-1]:.4f}, Training F1: {train_f1s[-1]:.4f}")
        print(f"Validation Loss: {val_losses[-1]:.4f}, Validation F1: {val_f1s[-1]:.4f}")

        if track_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
                "train_micro_f1": train_f1s[-1],
                "val_micro_f1": val_f1s[-1],
            })
        
        # early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            best_model_state = model.state_dict()  # save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state) # update model with best model state in case of overfittings
    model.eval()

    total_test_labels = []
    total_test_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Evaluation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
    
            valid_indices = labels != -100  # exclude padding tokens
            total_test_labels.extend(labels[valid_indices].cpu().numpy())
            total_test_preds.extend(torch.argmax(logits, dim=2)[valid_indices].cpu().numpy())
    
    # calculate micro and macro F1 scores (overall)
    test_micro_f1, test_macro_f1 = compute_f1(total_test_labels, total_test_preds)
    print(f"Test micro F1: {test_micro_f1}, Test macro F1: {test_macro_f1}")

    ################ outcomment this code if you do not want to calculate per class frequencies and correlations!
    # calculate the per class F1 (and the frequencies of the classes)
    per_class_f1 = compute_per_class_f1(total_test_labels, total_test_preds, label_map)
    print("Per-class F1 scores:")
    for label, scores in per_class_f1.items():
        print(f"{label}: F1={scores['f1']}, Support={scores['support']}")

    rows = []
    for label, scores in per_class_f1.items():
        rows.append({
            "label": label,
            "f1": scores["f1"],
            "support": scores["support"]
        })
    
    df = pd.DataFrame(rows) 
    filename = f"per_class_f1s_{model_name}_{seed}.csv"
    save_path = os.path.join(DRIVE_PLOT_DIR, filename)
    df.to_csv(save_path, index=False)
    
    # get the Pearson correlation between support (frequency) and the F1 score
    corr, p_value = pearsonr(df["support"], df["f1"])
    print(f"Pearson correlation between frequency and F1: {corr} (p={p_value})")
    
    # plot the relationship/correlation
    sns.scatterplot(data=df, x="support", y="f1")
    plt.title("Correlation between NE Class Frequency and F1 Score")
    plt.xlabel("Support (Frequency)")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.show()
    ####################
        
    
   # plot the confusion matrix
    conf_mat = confusion_matrix(np.array(total_test_labels).flatten(), np.array(total_test_preds).flatten())
    class_names = [label_map[i] for i in range(len(label_map))]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        conf_mat, 
        annot=True, 
        fmt='d', 
        cmap='BuPu', 
        xticklabels=class_names, 
        yticklabels=class_names, 
        annot_kws={"size": 8}
    )


    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    #plt.title(f'Confusion Matrix for {model_name}')
    plt.tight_layout()
    filename = f"confusion_matrix_{model_name}_{seed}.png"

    #save_path = os.path.join(RESULTS_DIR_NER, filename)
    save_path = os.path.join(DRIVE_PLOT_DIR, filename)
    plt.savefig(save_path , dpi=600)
    if track_wandb:
        wandb.log({"confusion_matrix": wandb.Image(save_path)})
        wandb.finish()
    plt.show()
    return model, test_micro_f1, test_macro_f1, train_losses, val_losses, train_f1s, val_f1s


def train_and_evaluate_NER_optuna(model_name, seed, train_dataloader, val_dataloader, test_dataloader, lr, weight_decay, num_epochs, dropout, num_labels, device, max_norm = 1.0, track_wandb=True):
    """Trains and evaluates NER models for an optuna study."""
    model = BertForTokenClassification.from_pretrained(MODEL_CONFIGS[model_name]["model_name"], num_labels=num_labels)
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model.to(device)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr schedule with warm up in the first 10% steps (cf. Gu et al. 2021)
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # early stopping
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []

    best_val_f1_micro = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_labels = []
        total_train_preds = []

        # cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #print(outputs[0]) # sanity check: verify whether loss is good 3.2689 -ln(1/27)
            logits = outputs.logits
            #print(logits.shape)# sanity check: (batch_size, sequence_length, num_labels) 

            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_labels.extend(labels.cpu().numpy())
            total_train_preds.extend(torch.argmax(logits, dim=2).cpu().numpy())

        train_losses.append(total_train_loss / len(train_dataloader))
        train_micro_f1, train_macro_f1 = compute_f1(total_train_labels, total_train_preds)
        train_f1s.append(train_micro_f1)

        # validation
        model.eval()
        total_val_labels = []
        total_val_preds = []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                total_val_loss += loss.item()
                total_val_labels.extend(labels.cpu().numpy())
                total_val_preds.extend(torch.argmax(logits, dim=2).cpu().numpy())

        val_losses.append(total_val_loss / len(val_dataloader))
        val_micro_f1, val_macro_f1 = compute_f1(total_val_labels, total_val_preds)
        val_f1s.append(val_micro_f1)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_losses[-1]:.4f}, Training F1: {train_f1s[-1]:.4f}")
        print(f"Validation Loss: {val_losses[-1]:.4f}, Validation F1: {val_f1s[-1]:.4f}")

        if val_micro_f1 > best_val_f1_micro:
            best_val_f1_micro = val_micro_f1

        if track_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
                "train_micro_f1": train_f1s[-1],
                "val_micro_f1": val_f1s[-1],
            })
        
        # early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return best_val_f1_micro


############## RE Training ###################


def train_and_evaluate_RE(model_name, tokenizer_voc_size, seed, train_dataloader, val_dataloader, test_dataloader, lr, weight_decay, num_epochs, dropout, device, max_norm, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, threshold,track_wandb=True):
    """
    Trains and evaluates a relation classification model.
    """
    
    model_str = MODEL_CONFIGS[model_name]["model_name"]
    model = AutoModel.from_pretrained(model_str)
    model.resize_token_embeddings(tokenizer_voc_size)  # adjust embeddings of the model for special tokens

    best_model_state = None # track best model state in case of overfitting

    hidden_size = model.config.hidden_size
    
    model = RelationClassifier_ent1_ent2_average_pooled(model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id).to(device) # CHANGE RELATIONCLASSIFER HERE

    bce_loss = nn.BCEWithLogitsLoss()  # use binary crossentropy loss, cf. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html (combines a Sigmoid layer and BCELoss).
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_training_steps = len(train_dataloader) * num_epochs # this will be displayed in wandb on the x-axis
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warm up steps
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    patience = 1 # 1 epoch patience
    patience_counter = 0

    train_losses, val_losses = [], []
    train_f1s_micro, val_f1s_micro = [], []
    train_f1s_macro, val_f1s_macro = [], []

    global_step = 0
    global_step_val = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        all_train_labels, all_train_preds = [], []

        # cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()

            optimizer.zero_grad() # zero parameter gradients
            
            outputs = model(input_ids, attention_mask) # forward pass
            loss = bce_loss(outputs, labels) # calculate bce loss
            if torch.isnan(loss):
                    print(f"Batch had nan loss, skipping this batch.")
                    continue
            loss.backward() # backward pass 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step() # optimize
            scheduler.step()

            total_train_loss += loss.item()
            
            preds = (torch.sigmoid(outputs).detach().cpu().numpy() > threshold).astype(int) 
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

            batch_f1_micro = f1_score(labels.cpu().numpy(), preds, average="micro") # calculate micro F1
            batch_f1_macro = f1_score(labels.cpu().numpy(), preds, average="macro") # calculate macro F1

            if track_wandb:
                # log the micro F1 and macro F1 into wandb
                wandb.log({
                    "step": global_step,
                    "train_loss_batch": loss.item(),
                    "train_f1_micro_batch": batch_f1_micro,
                    "train_f1_macro_batch": batch_f1_macro,
                })
                global_step += 1

        train_losses.append(total_train_loss / len(train_dataloader))
        train_f1_micro = f1_score(all_train_labels, all_train_preds, average="micro")
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro")
        train_f1s_micro.append(train_f1_micro)
        train_f1s_macro.append(train_f1_macro)

        if track_wandb:
            wandb.log({
            "train_loss": train_losses[-1],  
            "train_f1_micro": train_f1s_micro[-1],
            "train_f1_macro": train_f1s_macro[-1],})
            #}, step=epoch + 1) # log per epoch not step

        model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).float()

                outputs = model(input_ids, attention_mask)
                loss = bce_loss(outputs, labels)
                if torch.isnan(loss):
                    print("Batch had nan loss, skipping this batch.")
                    continue


                total_val_loss += loss.item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(torch.sigmoid(outputs).cpu().numpy() > threshold)

                global_step_val += 1

        val_losses.append(total_val_loss / len(val_dataloader))
        val_f1_micro = f1_score(all_val_labels, all_val_preds, average="micro")
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")
        val_f1s_micro.append(val_f1_micro)
        val_f1s_macro.append(val_f1_macro)

        if track_wandb:
            wandb.log({
            "step": global_step_val,
            "val_loss": val_losses[-1],  
            "val_f1_micro": val_f1s_micro[-1],
            "val_f1_macro": val_f1s_macro[-1],
            }) 

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_losses[-1]:.3f}, Train F1 macro: {train_f1s_macro[-1]:.3f},Train F1 micro: {train_f1s_micro[-1]:.3f} ")
        print(f"Validation Loss: {val_losses[-1]:.3f}, Val F1 macro: {val_f1s_macro[-1]:.3f}, Val F1 micro: {val_f1s_micro[-1]:.3f} ")

        # early stopping should be triggered if loss is not decreasing
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early, no improvement in decreasing loss!")
                break

    # load the best model (with lowest loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval() # set to evaluation mode

    # evalzte on test set
    all_test_labels, all_test_preds = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Evaluation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()

            logits = model(input_ids, attention_mask)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(torch.sigmoid(logits).cpu().numpy() > threshold)

    test_f1_micro = f1_score(all_test_labels, all_test_preds, average='micro')
    test_f1_macro = f1_score(all_test_labels, all_test_preds, average='macro')

    print(f"Test micro f1: {test_f1_micro:.4f}")
    print(f"Test macro f1: {test_f1_macro:.4f}")

    if track_wandb:
        wandb.log({
            "test_f1_micro": test_f1_micro,
            "test_f1_macro": test_f1_macro,
        })

        wandb.finish()

    return model, test_f1_micro, test_f1_macro, train_losses, val_losses, train_f1s_micro, train_f1s_macro, val_f1s_micro, val_f1s_macro


def train_and_evaluate_RE_optuna(model_name, tokenizer_voc_size, seed, train_dataloader, val_dataloader, test_dataloader, lr, weight_decay, num_epochs, dropout, device, max_norm, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, threshold,track_wandb=True):
    """
    Trains and evaluates a relation classification model. This can be used for an optuna study so best model state etc do not have to be specified and no models get saved.
    """
    #set_seed(seed) 
    
    model_str = MODEL_CONFIGS[model_name]["model_name"]
    model = AutoModel.from_pretrained(model_str)
    model.resize_token_embeddings(tokenizer_voc_size)  # adjust embeddings of the model for special tokens

    hidden_size = model.config.hidden_size
    
    model = RelationClassifier_CLS_ent1_ent2_avg_pooled(model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id).to(device)

    bce_loss = nn.BCEWithLogitsLoss()  # use binary crossentropy loss, cf. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html (combines a Sigmoid layer and BCELoss).
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_training_steps = len(train_dataloader) * num_epochs # this will be displayed in wandb on the x-axis
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warm up steps
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    patience = 2 # 2 epochs patience
    patience_counter = 0

    train_losses, val_losses = [], []
    train_f1s_micro, val_f1s_micro = [], []
    train_f1s_macro, val_f1s_macro = [], []

    global_step = 0
    global_step_val = 0

    best_val_micro_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        all_train_labels, all_train_preds = [], []

        # cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()

            optimizer.zero_grad() # zero parameter gradients
            
            outputs = model(input_ids, attention_mask) # forward pass
            loss = bce_loss(outputs, labels) # calculate bce loss

            loss.backward() # backward pass 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step() # optimize
            scheduler.step()

            total_train_loss += loss.item()
            
            preds = (torch.sigmoid(outputs).detach().cpu().numpy() > threshold).astype(int) 
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

            batch_f1_micro = f1_score(labels.cpu().numpy(), preds, average="micro") # calculate micro F1
            batch_f1_macro = f1_score(labels.cpu().numpy(), preds, average="macro") # calculate macro F1

            if track_wandb:
                # log the micro F1 and macro F1 into wandb
                wandb.log({
                    "step": global_step,
                    "train_loss_batch": loss.item(),
                    "train_f1_micro_batch": batch_f1_micro,
                    "train_f1_macro_batch": batch_f1_macro,
                })
                global_step += 1

        train_losses.append(total_train_loss / len(train_dataloader))
        train_f1_micro = f1_score(all_train_labels, all_train_preds, average="micro")
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro")
        train_f1s_micro.append(train_f1_micro)
        train_f1s_macro.append(train_f1_macro)

        if track_wandb:
            wandb.log({
            "train_loss": train_losses[-1],  
            "train_f1_micro": train_f1s_micro[-1],
            "train_f1_macro": train_f1s_macro[-1],})
            #}, step=epoch + 1) # log per epoch not step

        model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).float()

                outputs = model(input_ids, attention_mask)
                loss = bce_loss(outputs, labels)

                total_val_loss += loss.item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(torch.sigmoid(outputs).cpu().numpy() > threshold)

                global_step_val += 1

        val_losses.append(total_val_loss / len(val_dataloader))
        val_f1_micro = f1_score(all_val_labels, all_val_preds, average="micro")
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")
        val_f1s_micro.append(val_f1_micro)
        val_f1s_macro.append(val_f1_macro)

        if val_f1_micro > best_val_micro_f1:
            best_val_micro_f1 = val_f1_micro 

        if track_wandb:
            wandb.log({
            "step": global_step_val,
            "val_loss": val_losses[-1],  
            "val_f1_micro": val_f1s_micro[-1],
            "val_f1_macro": val_f1s_macro[-1],
            }) 

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_losses[-1]:.3f}, Train F1 macro: {train_f1s_macro[-1]:.3f},Train F1 micro: {train_f1s_micro[-1]:.3f} ")
        print(f"Validation Loss: {val_losses[-1]:.3f}, Val F1 macro: {val_f1s_macro[-1]:.3f}, Val F1 micro: {val_f1s_micro[-1]:.3f} ")

        # early stopping should be triggered if loss is not decreasing
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early, no improvement in decreasing loss!")
                break

    if track_wandb:
        wandb.finish()

    return best_val_micro_f1


####### TERNARY RE

def train_and_evaluate_RE_ternary(model_name, tokenizer_voc_size, seed, train_dataloader, val_dataloader, test_dataloader, lr, weight_decay, num_epochs, dropout, device, max_norm, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels, index_to_label, track_wandb=True):
    """
    Trains and evaluates a ternary relation classification model.
    """
    
    model_str = MODEL_CONFIGS[model_name]["model_name"]
    model = AutoModel.from_pretrained(model_str)
    model.resize_token_embeddings(tokenizer_voc_size)  # adjust embeddings of the model for special tokens

    best_model_state = None # track best model state in case of overfitting

    hidden_size = model.config.hidden_size
    
    model = RelationClassifier_ternary_ent1_ent2_average_pooled(model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels).to(device) # CHANGE RELATIONCLASSIFER HERE

    ce_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_training_steps = len(train_dataloader) * num_epochs # this will be displayed in wandb on the x-axis
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warm up steps
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    patience = 2 # 2 epochs patience
    patience_counter = 0

    train_losses, val_losses = [], []
    train_f1s_micro, val_f1s_micro = [], []
    train_f1s_macro, val_f1s_macro = [], []

    global_step = 0
    global_step_val = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        all_train_labels, all_train_preds = [], []

        # cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).long()
            #print(batch['label'].dtype)

            optimizer.zero_grad() # zero parameter gradients
            
            outputs = model(input_ids, attention_mask) # forward pass
            loss = ce_loss(outputs, labels) # calculate bce loss
            if torch.isnan(loss):
                    print(f"Batch had nan loss, skipping this batch. Number of unique labels = {torch.unique(labels).numel()}")
                    continue
            loss.backward() # backward pass 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step() # optimize
            scheduler.step()

            total_train_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

            batch_f1_micro = f1_score(labels.cpu().numpy(), preds, average="micro") # calculate micro F1
            batch_f1_macro = f1_score(labels.cpu().numpy(), preds, average="macro") # calculate macro F1

            if track_wandb:
                # log the micro F1 and macro F1 into wandb
                wandb.log({
                    "step": global_step,
                    "train_loss_batch": loss.item(),
                    "train_f1_micro_batch": batch_f1_micro,
                    "train_f1_macro_batch": batch_f1_macro,
                })
                global_step += 1

        train_losses.append(total_train_loss / len(train_dataloader))
        train_f1_micro = f1_score(all_train_labels, all_train_preds, average="micro")
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro")
        train_f1s_micro.append(train_f1_micro)
        train_f1s_macro.append(train_f1_macro)

        if track_wandb:
            wandb.log({
            "train_loss": train_losses[-1],  
            "train_f1_micro": train_f1s_micro[-1],
            "train_f1_macro": train_f1s_macro[-1],})
            #}, step=epoch + 1) # log per epoch not step

        model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).long()

                outputs = model(input_ids, attention_mask)
                loss = ce_loss(outputs, labels)
                if torch.isnan(loss):
                    print("Batch had nan loss, skipping this batch.")
                    continue


                total_val_loss += loss.item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

                global_step_val += 1

        val_losses.append(total_val_loss / len(val_dataloader))
        val_f1_micro = f1_score(all_val_labels, all_val_preds, average="micro")
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")
        val_f1s_micro.append(val_f1_micro)
        val_f1s_macro.append(val_f1_macro)

        if track_wandb:
            wandb.log({
            "step": global_step_val,
            "val_loss": val_losses[-1],  
            "val_f1_micro": val_f1s_micro[-1],
            "val_f1_macro": val_f1s_macro[-1],
            }) 

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_losses[-1]:.3f}, Train F1 macro: {train_f1s_macro[-1]:.3f},Train F1 micro: {train_f1s_micro[-1]:.3f} ")
        print(f"Validation Loss: {val_losses[-1]:.3f}, Val F1 macro: {val_f1s_macro[-1]:.3f}, Val F1 micro: {val_f1s_micro[-1]:.3f} ")

        # early stopping should be triggered if loss is not decreasing
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early, no improvement in decreasing loss!")
                break

    # load the best model (with lowest loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval() # set to evaluation mode

    # evalzte on test set
    all_test_labels, all_test_preds = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Evaluation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).long()

            logits = model(input_ids, attention_mask)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    test_f1_micro = f1_score(all_test_labels, all_test_preds, average='micro')
    test_f1_macro = f1_score(all_test_labels, all_test_preds, average='macro')

    print(f"Test micro f1: {test_f1_micro:.4f}")
    print(f"Test macro f1: {test_f1_macro:.4f}")

    if track_wandb:
        wandb.log({
            "test_f1_micro": test_f1_micro,
            "test_f1_macro": test_f1_macro,
        })

        wandb.finish()

    # calculate the per class F1 (and the frequencies of the classes)
    per_class_f1 = compute_per_class_f1(all_test_labels, all_test_preds, label_map_RE)
    print("Per-class F1 scores:")
    for label, scores in per_class_f1.items():
        print(f"{label}: F1={scores['f1']:.4f}, Support={scores['support']}")

    rows = []
    for label, scores in per_class_f1.items():
        rows.append({
            "label": label,
            "f1": scores["f1"],
            "support": scores["support"]
        })
    
    df = pd.DataFrame(rows) 
    filename = f"per_class_f1s_{model_name}_{seed}.csv"
    save_path = os.path.join(DRIVE_PLOT_DIR_RE, filename)
    df.to_csv(save_path, index=False)

    # plot the confusion matrix
    conf_mat = confusion_matrix(np.array(all_test_labels).flatten(), np.array(all_test_preds).flatten())
    class_names = [index_to_label[i] for i in range(len(index_to_label))]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        conf_mat, 
        annot=True, 
        fmt='d', 
        cmap='BuPu', 
        xticklabels=class_names, 
        yticklabels=class_names, 
        annot_kws={"size": 11}
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Predicted labels',fontsize=14)
    plt.ylabel('True labels', fontsize=14)
    plt.tight_layout()
    filename = f"confusion_matrix_{model_name}_{seed}.png"

    #save_path = os.path.join(RESULTS_DIR_NER, filename)
    save_path = os.path.join(DRIVE_PLOT_DIR_RE, filename)
    plt.savefig(save_path , dpi=600)
    plt.show()


    return model, test_f1_micro, test_f1_macro, train_losses, val_losses, train_f1s_micro, train_f1s_macro, val_f1s_micro, val_f1s_macro


def train_and_evaluate_RE_ternary_optuna(model_name, tokenizer_voc_size, seed, train_dataloader, val_dataloader, test_dataloader, lr, weight_decay, num_epochs, dropout, device, max_norm, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels ,track_wandb=False):
    """
    Trains and evaluates a relation classification model. This can be used for an optuna study so best model state etc do not have to be specified and no models get saved.
    """
    #set_seed(seed) #### TO DO: Das könnte man hier einmal alles auslagern und dann für NER und RE beides nutzen?
    
    model_str = MODEL_CONFIGS[model_name]["model_name"]
    model = AutoModel.from_pretrained(model_str)
    model.resize_token_embeddings(tokenizer_voc_size)  # adjust embeddings of the model for special tokens

    hidden_size = model.config.hidden_size
    
    model = RelationClassifier_ternary_ent1_ent2_average_pooled(model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels).to(device)
    ce_loss = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_training_steps = len(train_dataloader) * num_epochs # this will be displayed in wandb on the x-axis
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warm up steps
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    patience = 2 # 2 epochs patience
    patience_counter = 0

    train_losses, val_losses = [], []
    train_f1s_micro, val_f1s_micro = [], []
    train_f1s_macro, val_f1s_macro = [], []

    global_step = 0
    global_step_val = 0
    
    best_val_f1_micro = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        all_train_labels, all_train_preds = [], []

        # cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).long()

            optimizer.zero_grad() # zero parameter gradients
            
            outputs = model(input_ids, attention_mask) # forward pass
            loss = ce_loss(outputs, labels) # calculate ce loss

            loss.backward() # backward pass 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step() # optimize
            scheduler.step()

            total_train_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

            batch_f1_micro = f1_score(labels.cpu().numpy(), preds, average="micro") # calculate micro F1
            batch_f1_macro = f1_score(labels.cpu().numpy(), preds, average="macro") # calculate macro F1

            if track_wandb:
                # log the micro F1 and macro F1 into wandb
                wandb.log({
                    "step": global_step,
                    "train_loss_batch": loss.item(),
                    "train_f1_micro_batch": batch_f1_micro,
                    "train_f1_macro_batch": batch_f1_macro,
                })
                global_step += 1

        train_losses.append(total_train_loss / len(train_dataloader))
        train_f1_micro = f1_score(all_train_labels, all_train_preds, average="micro")
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro")
        train_f1s_micro.append(train_f1_micro)
        train_f1s_macro.append(train_f1_macro)

        if track_wandb:
            wandb.log({
            "train_loss": train_losses[-1],  
            "train_f1_micro": train_f1s_micro[-1],
            "train_f1_macro": train_f1s_macro[-1],})
            #}, step=epoch + 1) # log per epoch not step

        model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).long()

                outputs = model(input_ids, attention_mask)
                loss = ce_loss(outputs, labels)

                total_val_loss += loss.item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

                global_step_val += 1

        val_losses.append(total_val_loss / len(val_dataloader))
        val_f1_micro = f1_score(all_val_labels, all_val_preds, average="micro")
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")
        val_f1s_micro.append(val_f1_micro)
        val_f1s_macro.append(val_f1_macro)

        if track_wandb:
            wandb.log({
            "step": global_step_val,
            "val_loss": val_losses[-1],  
            "val_f1_micro": val_f1s_micro[-1],
            "val_f1_macro": val_f1s_macro[-1],
            }) 

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_losses[-1]:.3f}, Train F1 macro: {train_f1s_macro[-1]:.3f},Train F1 micro: {train_f1s_micro[-1]:.3f} ")
        print(f"Validation Loss: {val_losses[-1]:.3f}, Val F1 macro: {val_f1s_macro[-1]:.3f}, Val F1 micro: {val_f1s_micro[-1]:.3f} ")

        if val_f1_micro > best_val_f1_micro:
            best_val_f1_micro = val_f1_micro

        # early stopping should be triggered if loss is not decreasing
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Stopping early, no improvement in decreasing loss!")
                break
    if track_wandb: 
        wandb.finish()


    return best_val_f1_micro

###### References ###########

# Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., ... & Poon, H. 2021. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH), 3(1), 1-23.
# Chollet, F. 2018. Deep learning with Python. Manning Shelter Island.