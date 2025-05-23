
def get_entities(abstract_id, data):
    # get entities for that abstract id
    article_data = data.get(abstract_id, {})
    entities = article_data.get("entities", [])
    return entities

def get_entities_ground_truth(article_data):
    # article_data is already data[abstract_id]
    return article_data.get("entities", [])

def generate_candidate_pairs(entities):
    candidate_pairs = [] # get all possible candidate pairs
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j: # reflexive relationships not considered
                continue
            candidate_pairs.append((entities[i], entities[j])) # entities look like this: {'start_idx': 0, 'end_idx': 26, 'location': 'title', 'text_span': 'Lactobacillus fermentum NS9', 'label': 'dietary supplement'}
    return candidate_pairs
    
