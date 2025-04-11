
# TO DO: ADD EVERYTHING HERE

model = model.eval()
tokenizer = tokenizer 
THRESHOLD_INFERENCE = 0.85
THRESHOLD = 0.6
use_ground_truth = True

predictions = {}

# ground truth path for NERs 
GROUND_TRUTH_PATH = "/kaggle/input/gutbrainie2025/gutbrainie2025/Annotations/Dev/json_format/dev.json"

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)
    
def get_ground_truth_entities(abstract_id):
    # get entities for that abstract id
    article_data = ground_truth_data.get(abstract_id, {})
    entities = article_data.get("entities", [])
    return entities

def generate_candidate_pairs(entities):
    candidate_pairs = [] # get all possible candidate pairs
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j: # CHECK WHETHER THIS COULD TECHNICALLY BE POSSIBLE THAT SUBJ = OBJ, reflexive relationships?
                continue
            candidate_pairs.append((entities[i], entities[j])) # entities look like this: {'start_idx': 0, 'end_idx': 26, 'location': 'title', 'text_span': 'Lactobacillus fermentum NS9', 'label': 'dietary supplement'}
    return candidate_pairs
    
    
if use_ground_truth:
    for abstract_id, article_data in tqdm(ground_truth_data.items(), desc="Processing Abstracts", unit="abstract"):
        entities = get_ground_truth_entities(abstract_id)
        candidate_pairs = generate_candidate_pairs(entities)

        metadata = article_data.get("metadata", {})
        title = metadata.get("title", "")
        abstract = metadata.get("abstract", "")
        full_text = (title + " " + abstract).strip()  # get the combination of text and abstract
        offset = len(title) + 1 # offset for abstract positions
        
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

with open(f"relation_predictions_{model_name}.json", "w") as f:
    json.dump(predictions, f, indent=4)