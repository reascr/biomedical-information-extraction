import torch
import torch.nn as nn

######## NER #######################
# current code uses BERTForTokenClassification, replace with custom model if needed

######## RE ####################

class RelationClassifier_CLS_ent1_ent2_avg_pooled(nn.Module):
    def __init__(self, model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id):
        """
        Binary relation classification model using a BERT-based model with a linear classification layer on top.
        The hidden size should reflect the adjusted embedding size of the model after adding special entity tokens/markers to the tokenizer.
        """

        super(RelationClassifier_CLS_ent1_ent2_avg_pooled, self).__init__()
        self.transformer = model # BERT model 
        self.dropout = nn.Dropout(dropout) # add dropout
        self.hidden_size = hidden_size
        #self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 3, num_heads=8)
        self.classifier = nn.Linear(hidden_size * 3, 1) # input is concatenation of CLS + ent1 + ent2, output is one logit (binary classification)

        # ids of the entity markers
        self.ent1_start_id = ent1_start_id 
        self.ent1_end_id = ent1_end_id
        self.ent2_start_id = ent2_start_id
        self.ent2_end_id = ent2_end_id

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask) # get model outputs
        sequence_output = outputs.last_hidden_state  # take the last hidden state, shape: (batch_size, seq_len, hidden_size) 
        cls_repr = sequence_output[:, 0, :] # CLS token (first token in each sequence)
        
        batch_size = input_ids.size(0)
        ent1_repr_list = [] # store entity representations (one per sequence), list has length = batch size
        ent2_repr_list = [] 
        
        # iterate over sequences in a batch
        for i in range(batch_size):
            tokens = input_ids[i]  # shape: (seq_len)
            token_reps = sequence_output[i] # shape: (seq_len, hidden_size)
            
            # get the start and end of entities by the entity markers in input ids. The .nonzero() function finds the positions in the tensor where these tokens occur.
            ent1_start_pos = (tokens == self.ent1_start_id).nonzero(as_tuple=True)[0] # as_tuple=True to get 1D tensor. [0] takes the first occurrence of start token (since there is only one)
            ent1_end_pos = (tokens == self.ent1_end_id).nonzero(as_tuple=True)[0]
            ent2_start_pos = (tokens == self.ent2_start_id).nonzero(as_tuple=True)[0]
            ent2_end_pos = (tokens == self.ent2_end_id).nonzero(as_tuple=True)[0]
            
            # extract entity representations and average over subtokens
            if len(ent1_start_pos) > 0 and len(ent1_end_pos) > 0:
                start_idx = ent1_start_pos[0].item() + 1 #  convert 1D tensor to integer and exluce the ent marker pos (just take the average of tokens between them)
                end_idx = ent1_end_pos[0].item()
                ent1_repr = token_reps[start_idx:end_idx].mean(dim=0) # average over subtokens 
            else:
                ent1_repr = torch.zeros(token_reps.size(1), device=token_reps.device) # otherwise return 0 vector
            
            if len(ent2_start_pos) > 0 and len(ent2_end_pos) > 0:
                start_idx = ent2_start_pos[0].item() + 1
                end_idx = ent2_end_pos[0].item()
                ent2_repr = token_reps[start_idx:end_idx].mean(dim=0)  # average over subtokens 
            else:
                ent2_repr = torch.zeros(token_reps.size(1), device=token_reps.device)
                
            ent1_repr_list.append(ent1_repr) # add the entity representation for this sequence to the list for the whole batch
            ent2_repr_list.append(ent2_repr)
        
        ent1_repr = torch.stack(ent1_repr_list, dim=0) # stack the 16 (=batch size) tensors (with shape=hidden_size) along batch dimension =  --> shape = (batch_size,hidden_dim)
        ent2_repr = torch.stack(ent2_repr_list, dim=0)
        
        # concatentate CLS, entity1, and entity2 representations
        combined_repr = torch.cat([cls_repr, ent1_repr, ent2_repr], dim=1) # concatenate CLS token + entity representations along second dimension --> shape = (batch_size, hidden_dim*3) 
        combined_repr = self.dropout(combined_repr) # add a dropout layer
        
        # single logit as output (for binary classification)
        logit = self.classifier(combined_repr).squeeze(1)  # shape: (batch_size). We want only one logit per seq

        return logit 


class RelationClassifier_ent1_ent2_average_pooled(nn.Module):
    '''
    Passes averaged pooled entity 1 and entity 2 to the binary classifier.
    '''
    def __init__(self, model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id):
        super(RelationClassifier_ent1_ent2_average_pooled, self).__init__()
        self.transformer = model
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 1)  # only ent1 + ent2, no CLS

        self.ent1_start_id = ent1_start_id
        self.ent1_end_id = ent1_end_id
        self.ent2_start_id = ent2_start_id
        self.ent2_end_id = ent2_end_id

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        ent1_repr_list = []
        ent2_repr_list = []

        for i in range(batch_size):
            tokens = input_ids[i]
            token_reps = sequence_output[i]

            ent1_start_pos = (tokens == self.ent1_start_id).nonzero(as_tuple=True)[0]
            ent1_end_pos = (tokens == self.ent1_end_id).nonzero(as_tuple=True)[0]
            ent2_start_pos = (tokens == self.ent2_start_id).nonzero(as_tuple=True)[0]
            ent2_end_pos = (tokens == self.ent2_end_id).nonzero(as_tuple=True)[0]

            if len(ent1_start_pos) > 0 and len(ent1_end_pos) > 0:
                start_idx = ent1_start_pos[0].item() + 1
                end_idx = ent1_end_pos[0].item()
                ent1_repr = token_reps[start_idx:end_idx].mean(dim=0)
            else:
                ent1_repr = torch.zeros(token_reps.size(1), device=token_reps.device)

            if len(ent2_start_pos) > 0 and len(ent2_end_pos) > 0:
                start_idx = ent2_start_pos[0].item() + 1
                end_idx = ent2_end_pos[0].item()
                ent2_repr = token_reps[start_idx:end_idx].mean(dim=0)
            else:
                ent2_repr = torch.zeros(token_reps.size(1), device=token_reps.device)

            ent1_repr_list.append(ent1_repr)
            ent2_repr_list.append(ent2_repr)

        ent1_repr = torch.stack(ent1_repr_list, dim=0)
        ent2_repr = torch.stack(ent2_repr_list, dim=0)

        combined_repr = torch.cat([ent1_repr, ent2_repr], dim=1)  # no CLS
        combined_repr = self.dropout(combined_repr)

        logit = self.classifier(combined_repr).squeeze(1)

        return logit


class RelationClassifier_ent1_ent2_start_token(nn.Module):
    '''
    Passes entity start markers to a binary classifier.
    '''
    def __init__(self, model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id):
        super(RelationClassifier_ent1_ent2_start_token, self).__init__()
        self.transformer = model
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 1)  

        self.ent1_start_id = ent1_start_id
        self.ent1_end_id = ent1_end_id
        self.ent2_start_id = ent2_start_id
        self.ent2_end_id = ent2_end_id

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        ent1_repr_list = []
        ent2_repr_list = []

        for i in range(batch_size):
            tokens = input_ids[i]
            token_reps = sequence_output[i]

            ent1_start_pos = (tokens == self.ent1_start_id).nonzero(as_tuple=True)[0]
            ent2_start_pos = (tokens == self.ent2_start_id).nonzero(as_tuple=True)[0]

            if len(ent1_start_pos) > 0:
                idx = ent1_start_pos[0].item() 
                ent1_repr = token_reps[idx]
            else:
                ent1_repr = torch.zeros(token_reps.size(1), device=token_reps.device)

            if len(ent2_start_pos) > 0:
                idx = ent2_start_pos[0].item() 
                ent2_repr = token_reps[idx]
            else:
                ent2_repr = torch.zeros(token_reps.size(1), device=token_reps.device)

            ent1_repr_list.append(ent1_repr)
            ent2_repr_list.append(ent2_repr)

        ent1_repr = torch.stack(ent1_repr_list, dim=0)
        ent2_repr = torch.stack(ent2_repr_list, dim=0)

        combined_repr = torch.cat([ent1_repr, ent2_repr], dim=1)  # no CLS
        combined_repr = self.dropout(combined_repr)

        logit = self.classifier(combined_repr).squeeze(1)

        return logit


    
class RelationClassifier_CLSOnly(nn.Module):
    '''
    Passes only the [CLS] token representation to a binary classifier.
    '''
    def __init__(self, model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id):
        super(RelationClassifier_CLSOnly, self).__init__()
        self.transformer = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)  # using only CLS

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0(first token in each sequence)

        cls_output = self.dropout(cls_output)
        logit = self.classifier(cls_output).squeeze(1)

        return logit


        ##### ternary RE (multiclass classification)

class RelationClassifier_ternary_ent1_ent2_average_pooled(nn.Module):
    '''
    Passes averaged pooled entity 1 and entity 2 to a multiclass classifier.
    '''
    def __init__(self, model, hidden_size, dropout, ent1_start_id, ent1_end_id, ent2_start_id, ent2_end_id, num_labels):
        super(RelationClassifier_ternary_ent1_ent2_average_pooled, self).__init__()
        self.transformer = model
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # only ent1 + ent2, no CLS

        self.ent1_start_id = ent1_start_id
        self.ent1_end_id = ent1_end_id
        self.ent2_start_id = ent2_start_id
        self.ent2_end_id = ent2_end_id

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        ent1_repr_list = []
        ent2_repr_list = []

        for i in range(batch_size):
            tokens = input_ids[i]
            token_reps = sequence_output[i]

            ent1_start_pos = (tokens == self.ent1_start_id).nonzero(as_tuple=True)[0]
            ent1_end_pos = (tokens == self.ent1_end_id).nonzero(as_tuple=True)[0]
            ent2_start_pos = (tokens == self.ent2_start_id).nonzero(as_tuple=True)[0]
            ent2_end_pos = (tokens == self.ent2_end_id).nonzero(as_tuple=True)[0]

            if len(ent1_start_pos) > 0 and len(ent1_end_pos) > 0:
                start_idx = ent1_start_pos[0].item() + 1
                end_idx = ent1_end_pos[0].item()
                ent1_repr = token_reps[start_idx:end_idx].mean(dim=0)
            else:
                ent1_repr = torch.zeros(token_reps.size(1), device=token_reps.device)

            if len(ent2_start_pos) > 0 and len(ent2_end_pos) > 0:
                start_idx = ent2_start_pos[0].item() + 1
                end_idx = ent2_end_pos[0].item()
                ent2_repr = token_reps[start_idx:end_idx].mean(dim=0)
            else:
                ent2_repr = torch.zeros(token_reps.size(1), device=token_reps.device)

            ent1_repr_list.append(ent1_repr)
            ent2_repr_list.append(ent2_repr)

        ent1_repr = torch.stack(ent1_repr_list, dim=0)
        ent2_repr = torch.stack(ent2_repr_list, dim=0)

        combined_repr = torch.cat([ent1_repr, ent2_repr], dim=1)  # no CLS
        combined_repr = self.dropout(combined_repr)

        logit = self.classifier(combined_repr)

        return logit
