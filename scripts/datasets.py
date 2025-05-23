import os
import json
import re
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir,"..", "gutbrainie2025")


class AnnotationDataset(Dataset):
    def __init__(self, root_path, tokenizer=None, split='Train', quality_filter=['platinum_quality', 'gold_quality', 'silver_quality']):
        self.samples = []
        annotations_dir = os.path.join(root_path, 'Annotations', split)
            
        self.tokenizer = tokenizer
               
        if split == 'Train':
            for quality in quality_filter:  # filter out bronze quality since it contains autogenerated annotations
                quality_dir = os.path.join(annotations_dir, quality)
                json_format_dir = os.path.join(quality_dir, 'json_format')
                if not os.path.exists(json_format_dir):
                    print(f"No folder {json_format_dir} was found!")
                    continue
                
                # append data points (tuple of article identifier and corresponding annotations as a dictionary) to the sample list 
                for file_name in os.listdir(json_format_dir):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(json_format_dir, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            #self.samples.extend(data.items())  
                            sorted_items = sorted(data.items(), key=lambda item: item[0])  # sort items by article identifier number
                            self.samples.extend(sorted_items)
                          
        elif split == 'Dev':
            json_format_dir = os.path.join(annotations_dir, 'json_format')
            if not os.path.exists(json_format_dir):
                raise FileNotFoundError(f"No folder {json_format_dir} was found!")
                
            json_files = [fname for fname in os.listdir(json_format_dir) if fname.endswith('.json')]
            for json_file in json_files:
                file_path = os.path.join(json_format_dir, json_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sorted_items = sorted(data.items(), key=lambda item: item[0]) 
                    self.samples.extend(sorted_items)        
        else:
            raise ValueError("Specify a split, must be either 'Train' or 'Dev'!")
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]  # one data point (=article id) with annotations
        
    def plot_abstract_lengths(self):
        """
        Plots the distribution of tokenized word lengths of abstracts using either whitespace tokenization or BERT tokenization.
        """
        abstract_lengths = []
        for article_id, data in self.samples:
            abstract = data['metadata'].get('abstract', '')
            
            if self.tokenizer:  # use BERT tokenizer if its given
                tokens = self.tokenizer.tokenize(abstract)
                token_count = len(tokens)
                abstract_lengths.append(token_count)
                tokenizer_type = "BERT Tokenized"
            else:  # white space tokenization (just as an overview, baselines use NLTK tokenizer)
                word_count = len(abstract.split())
                abstract_lengths.append(word_count)
                tokenizer_type = "Whitespace Tokenized"
                
        print("Maximum number of tokens per abstract: ", max(abstract_lengths))
        plt.figure(figsize=(8, 4))
        plt.hist(abstract_lengths, bins=30, color='#E6E6FA', edgecolor='#D1C8E3')
        plt.title(f"Distribution of Abstract Lengths ({tokenizer_type})", fontsize=14, fontweight='bold')
        plt.xlabel("Token Count" if self.tokenizer else "Word Count", fontsize=12, fontweight='medium')
        plt.ylabel("Frequency", fontsize=12, fontweight='medium')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2, direction='in', grid_alpha=0.5)
    
        plt.tight_layout()
        plt.show()

    
    def get_text_data(self):
        """
        Extracts title and abstract text from the dataset.
        """
        all_titles = []
        all_abstracts = []
        
        for _, data in self.samples:
            if 'metadata' in data:
                if 'title' in data['metadata'] and data['metadata']['title']:
                    all_titles.append(data['metadata']['title'])
                if 'abstract' in data['metadata'] and data['metadata']['abstract']:
                    all_abstracts.append(data['metadata']['abstract'])
            
        return " ".join(all_titles), " ".join(all_abstracts)

    def build_vocab(self): # important for vocabulary coverage check 
        """
        Tokenizes the dataset and builds a vocabulary.
        """
        vocab = Counter()
        all_titles, all_abstracts = self.get_text_data()  # get the raw text
        
        # tokenize text (based on whitespace and punctuation)
        words = re.findall(r'\b\w+\b', all_titles.lower()) + re.findall(r'\b\w+\b', all_abstracts.lower())
        
        vocab.update(words)  # count all word occurences
        return vocab


def split_datasets(train_data, val_data, test_data):
    '''
    Splits the training dataset into a new training and validation set.  
    The validation set is sized based on the test set length.  
    '''
    train_size = len(train_data.samples)
    test_size = len(test_data.samples) 
    val_size = test_size  
    train_new_size = train_size - val_size  # remaining size for new training set

    train_subset, val_subset = random_split(train_data.samples, [train_new_size, val_size]) # split the train set into new train and val

    train_data.samples = train_subset
    val_data.samples = val_subset

    return train_data, val_data


def split_datasets_RE(train_data, val_data, test_data):
    '''
    Splits the training dataset into a new training and validation set.  
    The validation set is sized based on the test set length.  
    '''
    train_size = len(train_data.relation_samples)
    test_size = len(test_data.relation_samples) 
    val_size = test_size  
    train_new_size = train_size - val_size  # remaining size for new training set

    train_subset, val_subset = random_split(train_data.relation_samples, [train_new_size, val_size]) # split the train set into new train and val

    train_data.relation_samples = train_subset
    val_data.relation_samples = val_subset

    return train_data, val_data


############ NER class and helper functions ################

class NERDataset(AnnotationDataset):
    def __init__(self, root_path, tokenizer, max_length=512, split="Train", quality_filter=['platinum_quality', 'gold_quality', 'silver_quality']):
        """
        Creates a NER dataset.
        """
        super().__init__(root_path, tokenizer=tokenizer, split=split, quality_filter=quality_filter)
        #self.tokenizer = tokenizer
        self.max_length = max_length
        self.entity_classes = self.extract_entity_classes() # extract the entitiy classes in a BIO tagging scheme
        self.label_map = {label: idx for idx, label in enumerate(self.entity_classes)}

        # make title and abstract separate entries
        expanded_samples = []
        for article_id, data in self.samples: # articles are keys, data are values in the nested dictionary 
            if 'metadata' in data:
                if 'title' in data['metadata'] and data['metadata']['title']:
                    expanded_samples.append((article_id, data, "title"))
                if 'abstract' in data['metadata'] and data['metadata']['abstract']:
                    expanded_samples.append((article_id, data, "abstract"))
        
        self.samples = expanded_samples  # update samples with the new samples

    def extract_entity_classes(self):
        """
        Extract unique entity labels from the dataset for BIO tagging.
        """
        entity_labels = set()
        entity_labels.add("O")  # O (= Outside) is default label
        
        for _, data in self.samples:
            for entity in data.get('entities', []):
                entity_type = entity["label"]
                entity_labels.add(f"B-{entity_type}")
                entity_labels.add(f"I-{entity_type}")
        
        return sorted(list(entity_labels))

    def align_labels(self, text, entities, tokenized_input):
        """
        Convert character-level entity annotations to token-level BIO labels. 
        """
        labels = ["O"] * len(tokenized_input["input_ids"]) # O (= Outside) is default label

        subtokens = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
        
        for entity in entities:
           # print()
            start_idx, end_idx, label = entity["start_idx"], entity["end_idx"], entity["label"]
           # print(start_idx, end_idx, label)
            entity_tokens = self.tokenizer.encode(text[start_idx:end_idx+1], add_special_tokens=False) # tokenize the NE to assign subtokens labels
           # print(entity_tokens)
           # print(tokenizer.convert_ids_to_tokens(entity_tokens))
            offset_mapping = tokenized_input["offset_mapping"]

            try:
                start_index_label = next(i for i, (start, end) in enumerate(offset_mapping) if start == start_idx) # find the start...
            except: 
                break
                
            #print(start_index_label)
            
            labels[start_index_label] = f"B-{label}" 
            for j in range(1, len(entity_tokens)):
                if (start_index_label + j) < len(labels):
                    labels[start_index_label + j] = f"I-{label}"
    
            '''for i, token in enumerate(tokenized_input["input_ids"]):
                if token == entity_tokens[0]:  # start of entity --> B-tag
                    labels[i] = f"B-{label}" 
                    for j in range(1, len(entity_tokens)):
                        if (i + j) < len(labels):
                            labels[i + j] = f"I-{label}"'''
                                
            
        
        labels = [self.label_map[label] for label in labels] # map to numerical labels 
    
        # mask padding tokens (where attention_mask == 0) with -100 in the labels. This might also be circumvented by just masking out tokens with the attention mask?
        attention_mask = tokenized_input["attention_mask"]
        labels = [label if mask == 1 else -100 for label, mask in zip(labels, attention_mask)]
        
        return labels
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns tokenized data with entity labels and attention mask.
        """
        article_id, data, text_type = self.samples[idx]
        text = data['metadata'].get(text_type, '')
        entities = [e for e in data.get("entities", []) if e["location"] == text_type]
        
        tokenized_text = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        labels = self.align_labels(text, entities, tokenized_text)
        
        return {
            "input_ids": torch.tensor(tokenized_text["input_ids"]),
            "attention_mask": torch.tensor(tokenized_text["attention_mask"]),
            "labels": torch.tensor(labels),
        }

############ RE class and helper functions #################


# helper functions for modularity of code
def get_adjusted_indices(entity, offset):
    """
    Adjusts start and end indices for entities in the abstract (adds an offset = length of title to abstracts).
    """
    if entity.get("location") == "abstract":
        start_idx = entity["start_idx"] + offset
        end_idx = entity["end_idx"] + 1 + offset
    else:
        start_idx = entity["start_idx"]
        end_idx = entity["end_idx"] + 1
    return start_idx, end_idx


def mark_entities(text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx):
    """
    Inserts special entity markers into the text.
    """
    text_chars = list(text)
    text_chars.insert(obj_end_idx, "</ent2>")
    text_chars.insert(obj_start_idx, "<ent2>")
    text_chars.insert(subj_end_idx, "</ent1>")
    text_chars.insert(subj_start_idx, "<ent1>")
    return "".join(text_chars)


class REDataset(AnnotationDataset):
    def __init__(self, root_path, tokenizer, max_length=512, split="Train", quality_filter=['platinum_quality', 'gold_quality', 'silver_quality']):
        """
        Creates a relation extraction dataset.
        Each data point is a concatenation of abstract and title. Entity markers (<ent1> and <ent2>) are inserted to mark the two entities in question.
        These entity markers have to be added to the tokenizer that is passed to the initialisation of the class.
        For each article, positive relation candidates are generated based on the ground truth. For this, all possible mention based relations are considered.
        (A set of tag based relations is later inferred during inference. Here, the entities are not modified except for a special entity marker).
        Negative samples are generated from other candidate entity pairs (randomly, check later to include easy, medium, and hard examples, especially entities that are the same ones as in the relations).
        """
        super().__init__(root_path, tokenizer=tokenizer, split=split, quality_filter=quality_filter)
        #self.tokenizer = tokenizer # is already initiated by the parent class Annotation Data set
        self.max_length = max_length
        self.relation_samples = []

        #counter = 0
        
        # concatenate title and abstract because a relation can hold between an entity in the title and one in the abstract
        for article_id, data in self.samples:
            #counter += 1
            title = data['metadata'].get('title', '')
            abstract = data['metadata'].get('abstract', '')
            full_text = (title + " " + abstract).strip() 
            if not full_text:
                continue
            
            # get ground truth entities and all relations 
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # for entities in the abstract, we need to add the length of the title to the indices (since we are concatenating titles and abstracts)
            offset = len(title) + 1

            for rel in relations:
                subj_entity = {"start_idx": rel["subject_start_idx"], "end_idx": rel["subject_end_idx"], "location": rel["subject_location"]}
                subj_start_idx, subj_end_idx = get_adjusted_indices(subj_entity, offset) # adjust indices for subject
    
                obj_entity = {"start_idx": rel["object_start_idx"], "end_idx": rel["object_end_idx"], "location": rel["object_location"]}
                obj_start_idx, obj_end_idx = get_adjusted_indices(obj_entity, offset) # adjust indices for object

                marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    
                self.relation_samples.append({
                    "article_id": article_id,
                    "text": marked_text,
                    "label": 1 # positive relation
                })

            # create negative examples by generating as many negative examples as positives (to balance classes)
            num_pos = len(relations)
            #if counter < 2:
                #print(num_pos)

            candidate_pairs = [] # get all possible candidate pairs
            for i in range(len(entities)):
                for j in range(len(entities)):
                    candidate_pairs.append((entities[i], entities[j])) # entities look like this: {'start_idx': 0, 'end_idx': 26, 'location': 'title', 'text_span': 'Lactobacillus fermentum NS9', 'label': 'dietary supplement'}
            
            random.shuffle(candidate_pairs)
            negatives_added = 0
            # random sampling of negatives
            for pair in candidate_pairs:
                subj, obj = pair
                # exclude all possible candidate pairs (order matters because we have directional relationships between subj and obj)
                is_positive = any(subj["text_span"] == r["subject_text_span"] and obj["text_span"] == r["object_text_span"] for r in relations)
                if not is_positive:
                    # add the negative example
                    subj_start_idx, subj_end_idx = get_adjusted_indices(subj, offset)
                    obj_start_idx, obj_end_idx = get_adjusted_indices(obj, offset)

                    marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    
                    self.relation_samples.append({
                        "article_id": article_id,
                        "text": marked_text,
                        "label": 0
                    })
                    
                    negatives_added += 1
                    if negatives_added >= num_pos:
                        break # we want a balanced data set, stop if number of positives is reached
            
        #random.shuffle(self.relation_samples)
        #self.relation_samples = self.relation_samples[:100] # for testing purposes
    def __len__(self):
        return len(self.relation_samples)
    
    def __getitem__(self, idx):
        """
        Returns a tokenized relation extraction data point:
            - input_ids
            - attention_mask
            - label (0 or 1) for binary classification
        """
        sample = self.relation_samples[idx]
        
        tokenized_text = self.tokenizer(
            sample["text"], 
            #padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
    
        for key in tokenized_text:
            tokenized_text[key] = tokenized_text[key].squeeze(0)
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }



class REDataset_enhanced_negative_sampling_strategy(AnnotationDataset):
    def __init__(self, root_path, tokenizer, max_length=512, split="Train", quality_filter=['platinum_quality', 'gold_quality', 'silver_quality']):
        """
        Creates a relation extraction dataset.
        Each data point is a concatenation of abstract and title. Entity markers (<ent1> and <ent2>) are inserted to mark the two entities in question.
        These entity markers have to be added to the tokenizer that is passed to the initialisation of the class.
        For each article, positive relation candidates are generated based on the ground truth. For this, all possible mention based relations are considered.
        (A set of tag based relations is later inferred during inference. Here, the entities are not modified except for a special entity marker).
        Negative samples are generated from other candidate entity pairs (randomly, check later to include easy, medium, and hard examples, especially entities that are the same ones as in the relations).
        """
        super().__init__(root_path, tokenizer=tokenizer, split=split, quality_filter=quality_filter)
        #self.tokenizer = tokenizer # is already initiated by the parent class Annotation Data set
        self.max_length = max_length
        self.relation_samples = []

        # concatenate title and abstract because a relation can hold between an entity in the title and one in the abstract
        for article_id, data in self.samples:
            title = data['metadata'].get('title', '')
            abstract = data['metadata'].get('abstract', '')
            full_text = (title + " " + abstract).strip() 
            if not full_text:
                continue
            
            # get ground truth entities and all relations 
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # for entities in the abstract, we need to add the length of the title to the indices (since we are concatenating titles and abstracts)
            offset = len(title) + 1

            reversed_pairs = [] # we need this for sampling of negatives below
            
            for rel in relations:
                subj_entity = {"start_idx": rel["subject_start_idx"], "end_idx": rel["subject_end_idx"], "location": rel["subject_location"]}
                subj_start_idx, subj_end_idx = get_adjusted_indices(subj_entity, offset) # adjust indices for subject
    
                obj_entity = {"start_idx": rel["object_start_idx"], "end_idx": rel["object_end_idx"], "location": rel["object_location"]}
                obj_start_idx, obj_end_idx = get_adjusted_indices(obj_entity, offset) # adjust indices for object

                
                reversed_pairs.append((obj_entity, subj_entity)) # reversed pairs

                marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    
                self.relation_samples.append({
                    "article_id": article_id,
                    "text": marked_text,
                    "label": 1 # positive relation
                })

            # create negative examples by generating as many negative examples as positives (to balance classes)
            num_pos = len(relations)
            num_reversed = int(num_pos*0.25)
            num_same_tag = int(num_pos*0.25)
            num_random = num_pos - num_reversed - num_same_tag
                
            candidate_pairs = [] # get all possible candidate pairs
            for i in range(len(entities)):
                for j in range(len(entities)):
                    candidate_pairs.append((entities[i], entities[j])) # entities look like this: {'start_idx': 0, 'end_idx': 26, 'location': 'title', 'text_span': 'Lactobacillus fermentum NS9', 'label': 'dietary supplement'}
            random.shuffle(candidate_pairs)
            
            same_tag_pairs = []
            random_pairs = []
            
            for subj, obj in candidate_pairs:
                # exclude all possible candidate pairs (order matters because we have directional relationships between subj and obj)
                is_positive = any(subj["text_span"] == r["subject_text_span"] and obj["text_span"] == r["object_text_span"] for r in relations)
                # exclude all pertubations of positive pairs (we subsample them above)
                is_reversed = any(obj["text_span"] == r["subject_text_span"] and subj["text_span"] == r["object_text_span"] for r in relations)
                has_same_tag = any(subj["label"] == r["subject_label"] and obj["label"] == r["object_label"] for r in relations)
                
                if not is_positive and not is_reversed:
                    if has_same_tag:
                        same_tag_pairs.append((subj,obj))
                    else:
                        random_pairs.append((subj,obj))

            # create random subset of the negatives
            random.shuffle(same_tag_pairs)
            random.shuffle(random_pairs)
            random.shuffle(reversed_pairs)

            sampled_reversed = reversed_pairs[:min(num_reversed, len(reversed_pairs))]
            sampled_same_tag = same_tag_pairs[:min(num_same_tag, len(same_tag_pairs))]
            sampled_random = random_pairs[:num_pos - len(sampled_reversed) - len(sampled_same_tag)]
            final_negative_pairs = sampled_reversed + sampled_same_tag + sampled_random

            for subj, obj in final_negative_pairs:
                    subj_start_idx, subj_end_idx = get_adjusted_indices(subj, offset)
                    obj_start_idx, obj_end_idx = get_adjusted_indices(obj, offset)
                    marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    self.relation_samples.append({
                        "article_id": article_id,
                        "text": marked_text,
                        "label": 0
                    }) # might store type of negative here for further examination

        #random.shuffle(self.relation_samples)
        #self.relation_samples = self.relation_samples[:100]
    
    def __len__(self):
        return len(self.relation_samples)
    
    def __getitem__(self, idx):
        """
        Returns a tokenized relation extraction data point:
            - input_ids
            - attention_mask
            - label (0 or 1) for binary classification
        """
        sample = self.relation_samples[idx]
        
        tokenized_text = self.tokenizer(
            sample["text"], 
            #padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
    
        for key in tokenized_text:
            tokenized_text[key] = tokenized_text[key].squeeze(0) # ???
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }




####### ternary RE #############

class REDataset_ternary(AnnotationDataset):
    def __init__(self, root_path, tokenizer, max_length=512, split="Train", quality_filter=['platinum_quality', 'gold_quality', 'silver_quality']):
        """
        Creates a relation extraction dataset.
        Each data point is a concatenation of abstract and title. Entity markers (<ent1> and <ent2>) are inserted to mark the two entities in question.
        These entity markers have to be added to the tokenizer that is passed to the initialisation of the class.
        For each article, positive relation candidates are generated based on the ground truth. For this, all possible mention based relations are considered.
        (A set of tag based relations is later inferred during inference. Here, the entities are not modified except for a special entity marker).
        Negative samples are generated from other candidate entity pairs (randomly, check later to include easy, medium, and hard examples, especially entities that are the same ones as in the relations).
        """
        super().__init__(root_path, tokenizer=tokenizer, split=split, quality_filter=quality_filter)
        #self.tokenizer = tokenizer # is already initiated by the parent class Annotation Data set
        self.max_length = max_length
        self.relation_samples = []

        # concatenate title and abstract because a relation can hold between an entity in the title and one in the abstract
        for article_id, data in self.samples:
            title = data['metadata'].get('title', '')
            abstract = data['metadata'].get('abstract', '')
            full_text = (title + " " + abstract).strip() 
            if not full_text:
                continue
            
            # get ground truth entities and all relations 
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # for entities in the abstract, we need to add the length of the title to the indices (since we are concatenating titles and abstracts)
            offset = len(title) + 1

            for rel in relations:
                subj_entity = {"start_idx": rel["subject_start_idx"], "end_idx": rel["subject_end_idx"], "location": rel["subject_location"]}
                subj_start_idx, subj_end_idx = get_adjusted_indices(subj_entity, offset) # adjust indices for subject
    
                obj_entity = {"start_idx": rel["object_start_idx"], "end_idx": rel["object_end_idx"], "location": rel["object_location"]}
                obj_start_idx, obj_end_idx = get_adjusted_indices(obj_entity, offset) # adjust indices for object

                predicate = rel["predicate"] # relation type

                marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    
                self.relation_samples.append({
                    "article_id": article_id,
                    "text": marked_text,
                    "label": predicate # e.g. "influence"
                })

            # create negative examples by generating as many negative examples as positives (to balance classes)
            num_pos = len(relations)

            candidate_pairs = [] # get all possible candidate pairs
            for i in range(len(entities)):
                for j in range(len(entities)):
                    candidate_pairs.append((entities[i], entities[j])) # entities look like this: {'start_idx': 0, 'end_idx': 26, 'location': 'title', 'text_span': 'Lactobacillus fermentum NS9', 'label': 'dietary supplement'}
            
            random.shuffle(candidate_pairs)
            negatives_added = 0
            # random sampling of negatives
            for pair in candidate_pairs:
                subj, obj = pair
                # exclude all possible candidate pairs (order matters because we have directional relationships between subj and obj)
                is_positive = any(subj["text_span"] == r["subject_text_span"] and obj["text_span"] == r["object_text_span"] for r in relations)
                if not is_positive:
                    subj_start_idx, subj_end_idx = get_adjusted_indices(subj, offset)
                    obj_start_idx, obj_end_idx = get_adjusted_indices(obj, offset)

                    marked_text = mark_entities(full_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
                    
                    self.relation_samples.append({
                        "article_id": article_id,
                        "text": marked_text,
                        "label": "no relation" # no relation
                    })
                    
                    negatives_added += 1
                    if negatives_added >= num_pos:
                        break # we want a balanced data set, stop if number of positives is reached

        all_labels = {s["label"] for s in self.relation_samples} # get a set of all possible labels, should be 17+1 (O label)
        # get no relation as 0 index
        other_labels = sorted(all_labels - {"no relation"}) 
        #self.label2id = {"no relation": 0}  
        #self.label2id.update({label: index + 1 for index, label in enumerate(other_labels)})  
        self.label2id = {label: idx for idx, label in enumerate(other_labels)}
        self.label2id["no relation"] = len(other_labels)
        self.id2label = {idx: lbl for lbl, idx in self.label2id.items()}
        for sample in self.relation_samples:
            sample["label"] = self.label2id[sample["label"]] # bring to numerical form
            
        #random.shuffle(self.relation_samples)

    def __len__(self):
        return len(self.relation_samples)
    
    def __getitem__(self, idx):
        """
        Returns a tokenized relation extraction data point:
            - input_ids
            - attention_mask
            - label for multiclass classification
        """
        sample = self.relation_samples[idx]
        
        tokenized_text = self.tokenizer(
            sample["text"], 
            #padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
    
        for key in tokenized_text:
            tokenized_text[key] = tokenized_text[key].squeeze(0)
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }


####### collate function and create dataloaders

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    
    # dynamic padding to longest seq of batch (to increase computational efficiency)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0) # tokenizer.pad_token_id
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0) # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    # keep batch size as first dimension. Tensor of size B x T x * where T is the length of the longest sequence
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "label": labels
    }



def create_dataloaders(batch_size, tokenizer, device):
    train_dataset = NERDataset(DATA_DIR, tokenizer=tokenizer, split="Train")
    val_dataset = NERDataset(DATA_DIR, tokenizer=tokenizer, split="Train")  # dummy val data set
    test_dataset = NERDataset(DATA_DIR, tokenizer=tokenizer, split="Dev")  # take test as val 

    # split into train and val 
    train_dataset, val_dataset = split_datasets(train_dataset, val_dataset, test_dataset)

    # create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True) 
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False) 
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader


def create_dataloaders_RE(batch_size, tokenizer, device): 
    train_dataset = REDataset(DATA_DIR, tokenizer=tokenizer, split="Train")
    val_dataset = REDataset(DATA_DIR, tokenizer=tokenizer, split="Train")  # dummy val data set
    test_dataset = REDataset(DATA_DIR, tokenizer=tokenizer, split="Dev")  # take dev set as test set (until official test set release)

    # split into train and val 
    train_dataset, val_dataset = split_datasets_RE(train_dataset, val_dataset, test_dataset)
    #print(train_dataset[1])
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader, test_dataloader


def create_dataloaders_RE_ternary(batch_size, tokenizer, device): 
    train_dataset = REDataset_ternary(DATA_DIR, tokenizer=tokenizer, split="Train")
    val_dataset = REDataset_ternary(DATA_DIR, tokenizer=tokenizer, split="Train")  # dummy val data set
    test_dataset = REDataset_ternary(DATA_DIR, tokenizer=tokenizer, split="Dev")  # take dev set as test set (until official test set release)

    # split into train and val 
    train_dataset, val_dataset = split_datasets_RE(train_dataset, val_dataset, test_dataset)
    #print(train_dataset[1])

    num_labels = len(train_dataset.label2id)
    index_to_label = train_dataset.id2label   
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader, test_dataloader, num_labels, index_to_label