# Very simple vanilla model
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embed_dim, sequence_length, num_classes, dropout=0.1):
        super(MLP, self).__init__()
            
        '''
        self.layers = nn.Sequential( 
            nn.Linear(embed_dim * sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        '''
        self.layers = nn.Sequential( 
            nn.Linear(embed_dim * sequence_length, num_classes),
        )
    def forward(self, _, x, attention_mask=None):
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)
        return x