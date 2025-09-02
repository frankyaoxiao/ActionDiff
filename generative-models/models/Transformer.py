import torch
import torch.nn as nn
import torch.nn.functional as F
from .TransformerDecoder import TransformerDecoder
from transformers import CLIPTokenizer, CLIPTextModel, logging
import open_clip

class ClassificationTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=1280, encoder_heads=8, encoder_layers=6, dropout=0.1):
        super(ClassificationTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=encoder_heads,
            dim_feedforward=3*embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=encoder_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, _, x, attention_mask=None):
        batch_size = x.shape[0]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_encoder(x)
        
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            extended_attention_mask = torch.cat((cls_mask, attention_mask), dim=1)
            src_key_padding_mask = (extended_attention_mask == 0)
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.encoder(x)
        
        cls_representation = x[:, 0]
        
        out = self.classifier(cls_representation)
        
        return out

def get_text_features(cl_list):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    texts = tokenizer(cl_list, padding=True, return_tensors="pt")
    text_class = text_model(**texts).pooler_output.detach()
    return text_class

class PositionalEncoding(nn.Module):
    """
    Classic sin/cos positional encoding
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out

