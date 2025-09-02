import torch
import torch.nn as nn
import torch.nn.functional as F
from .TransformerDecoder import TransformerDecoder
from transformers import CLIPTokenizer, CLIPTextModel, logging
import open_clip
from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2

class MSQNetWithoutImageCLIP(nn.Module):
    def __init__(self, class_embed, decoder_heads=8, decoder_layers=6):
        super(MSQNetWithoutImageCLIP, self).__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.hidden_size = 1280 
        self.linear = nn.Linear(self.hidden_size, self.embed_dim, bias=False)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=decoder_heads)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=decoder_norm, residual=False)
        self.transformer = nn.Transformer(d_model=self.embed_dim, custom_decoder=decoder)
        self.pos_encoder = PositionalEncoding(self.embed_dim, dropout=0.1)
        self.query_embed = nn.Parameter(class_embed)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, _, features, attention_mask=None):
        x = self.linear(features)
        query_embed = self.query_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            hs = self.transformer(x, query_embed, src_key_padding_mask=src_key_padding_mask)
        else:
            hs = self.transformer(x, query_embed)
        
        hs = hs.transpose(0, 1)
        out = self.group_linear(hs)
        return out

def get_text_features(cl_list):
   # cond_model = FrozenOpenCLIPEmbedder2(always_return_pooled=True, legacy=False).to('cuda')
   # with torch.cuda.amp.autocast(enabled=False): 
   #     _, text_class = cond_model(cl_list)
   # return text_class
   text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
   tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
   texts = tokenizer(cl_list, padding=True, return_tensors="pt")
   text_class = text_model(**texts).pooler_output.detach()
   return text_class.detach()

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.sqrt(torch.tensor(self.W.size(2)).float())
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

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