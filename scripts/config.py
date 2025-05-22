MODEL_CONFIGS = {
    "PubMedBERT": {
        "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", # uncased, no cased version available
        "tokenizer": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    },
    
    "BERT": {
        "model_name": "google-bert/bert-base-cased", # cased
        "tokenizer": "google-bert/bert-base-cased"
    },
    
    "BioBERT": {
        "model_name": "dmis-lab/biobert-v1.1", # cased
        "tokenizer": "dmis-lab/biobert-v1.1"
    }
}

'''
MODEL_CONFIGS = {
    "PubMedBERT": {
        "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", # uncased, no cased version available
        "tokenizer": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    },}
'''


label_map = {
    0: "B-DDF", 1: "B-anatomical location", 2: "B-animal", 3: "B-bacteria", 4: "B-biomedical technique",
    5: "B-chemical", 6: "B-dietary supplement", 7: "B-drug", 8: "B-food", 9: "B-gene", 10: "B-human",
    11: "B-microbiome", 12: "B-statistical technique", 13: "I-DDF", 14: "I-anatomical location",
    15: "I-animal", 16: "I-bacteria", 17: "I-biomedical technique", 18: "I-chemical",
    19: "I-dietary supplement", 20: "I-drug", 21: "I-food", 22: "I-gene", 23: "I-human",
    24: "I-microbiome", 25: "I-statistical technique", 26: "O"
}


label_map_RE = {0: 'administered', 1: 'affect', 2: 'change abundance', 3: 'change effect', 4: 'change expression', 5: 'compared to', 6: 'impact', 7: 'influence', 8: 'interact', 9: 'is a', 10: 'is linked to', 11: 'located in', 12: 'part of', 13: 'produced by', 14: 'strike', 15: 'target', 16: 'used by', 17: 'no relation'}


seeds = [42,1234,29,7,123]
#seeds = [42] # test