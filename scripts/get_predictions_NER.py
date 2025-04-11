import torch

def extract_entities(text, input_ids, predictions, offset_mapping, label_map, location):
    """
    Converts token-level predictions into entity spans.
    """
    entities = []
    current_entity = None

    for token_id, pred_label_id, offset in zip(input_ids, predictions, offset_mapping):
        if offset == [0, 0]: # exclude special tokens (they have an offset of [0,0])
            continue

        label = label_map.get(pred_label_id, "O")
        
        if label == "O": # close current entity if the label i O
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        if label.startswith("B-"): # start new entity if the label is the beginning of an entity
            if current_entity is not None:
                entities.append(current_entity)
            entity_label = label[2:]  # remove the "B-"
            current_entity = {
                "start_idx": offset[0],
                "end_idx": offset[1]-1,
                "location": location,
                "text_span": text[offset[0]:offset[1]], # always substract -1 from offset since in offset mapping it's excluding but in the ground truth annotation it's including
                "label": entity_label
            }

        elif label.startswith("I-"): # if the label starts with I, extend the current entity
            if current_entity is not None and current_entity["label"] == label[2:]: # do we have a current entity and the labels match?
                #extend the entity span
                current_entity["end_idx"] = offset[1]-1
                current_entity["text_span"] = text[current_entity["start_idx"]:offset[1]]
            else:
                entity_label = label[2:] # if there is no current entity: beginning tag
                current_entity = {
                    "start_idx": offset[0],
                    "end_idx": offset[1]-1,
                    "location": location,
                    "text_span": text[offset[0]:offset[1]],
                    "label": entity_label
                }
    # if there's an unfinished entity at the end, append this
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities

    

def predict_entities(text, location, model, tokenizer, device, label_map):
    '''Tokenizes text, runs the model prediction, and extracts entities.'''
    # tokenize the text with offsets (so we know the original character positions)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    input_ids_tensor = tokens["input_ids"].to(device)
    attention_mask_tensor = tokens["attention_mask"].to(device)

    offset_mapping = tokens["offset_mapping"].squeeze().tolist()
    
    with torch.no_grad():
        outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)
    logits = outputs.logits
    # get predicted label for each token
    predictions = torch.argmax(logits, dim=2).squeeze().tolist() # Also, take into account that we have to run predictions over 5 random seeds!!!
    input_ids = tokens["input_ids"].squeeze().tolist()
    entities = extract_entities(text, input_ids, predictions, offset_mapping, label_map, location) # extract the entities
    return entities
