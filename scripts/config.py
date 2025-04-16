MODEL_CONFIGS = {
    "PubMedBERT": {
        "model_name": "NeuML/pubmedbert-base-embeddings",
        "tokenizer": "NeuML/pubmedbert-base-embeddings"
    },
    
    "BERT": {
        "model_name": "bert-base-uncased",
        "tokenizer": "bert-base-uncased"
    },
    "BioBERT": {
        "model_name": "dmis-lab/biobert-v1.1",
        "tokenizer": "dmis-lab/biobert-v1.1"
    }
}

label_map = {
    0: "B-DDF", 1: "B-anatomical location", 2: "B-animal", 3: "B-bacteria", 4: "B-biomedical technique",
    5: "B-chemical", 6: "B-dietary supplement", 7: "B-drug", 8: "B-food", 9: "B-gene", 10: "B-human",
    11: "B-microbiome", 12: "B-statistical technique", 13: "I-DDF", 14: "I-anatomical location",
    15: "I-animal", 16: "I-bacteria", 17: "I-biomedical technique", 18: "I-chemical",
    19: "I-dietary supplement", 20: "I-drug", 21: "I-food", 22: "I-gene", 23: "I-human",
    24: "I-microbiome", 25: "I-statistical technique", 26: "O"
}

seeds = [42,1234,29,7,123]
#seeds = [1] # test 