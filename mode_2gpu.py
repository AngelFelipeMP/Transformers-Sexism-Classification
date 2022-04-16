import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransforomerModel(nn.Module):
    def __init__(self, transformer, drop_out, number_of_classes):
        super(TransforomerModel, self).__init__()
        self.number_of_classes = number_of_classes
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        
        # Embedding Layer --> cuda : 0
        self.embedding = self.transformer.embeddings.to('cuda:0')
        
        # Encoder Layer --> cuda : 1
        self.encoder = self.transformer.encoder.to('cuda:1')
        
        # Classifer --> cuda : 1
        self.dropout = nn.Dropout(drop_out).to('cuda:1')
        self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes).to('cuda:1')
        
    def forward(self, iputs):
        # Pass the input_ids to cuda:0 since embedding layer in cuda:0
        emb_out = self.embedding(**iputs.to('cuda:0'))
        
        # Move the outputs of embedding layer to cuda:1 as input to encoder layer
        enc_out = self.encoder(emb_out.to('cuda:1'))
        
        mean_pool = torch.mean(enc_out['last_hidden_state'], 1)
        max_pool, _ = torch.max(enc_out['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)

        return self.classifier(drop)