import torch
import torch.nn as nn
import numpy as np
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        self.nhead = cfg['nhead']
        self.attention = nn.MultiheadAttention(cfg['d_model'], self.nhead, batch_first=True)

    def forward(self, q, k, v, attn_mask=None):
        return self.attention(q, k, v, key_padding_mask=attn_mask)[0]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(cfg['d_model'], cfg['d_ff'], bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg['d_ff'], cfg['d_model'], bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(cfg)
        self.position_wise_feed_forward = PositionWiseFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg['d_model'])
        self.norm2 = nn.LayerNorm(cfg['d_model'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x, attn_mask):
        attn_output = self.multi_head_attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.position_wise_feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']

        self.embed = nn.Linear(cfg['n_features'], self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg['num_layers'])])
        
    def forward(self, x, attn_mask=None):
        self.batch_size, self.seq_len, self.n_features = x.shape
        
        x = self.embed(x)
        x = x + self.get_position_encoding(self.seq_len, self.d_model).to(self.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x
    
    def get_position_encoding(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        pos_enc = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                theta = get_angle(pos, i, d_model)
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(theta)
                else:
                    pos_enc[pos, i] = np.cos(theta)
        return torch.FloatTensor(pos_enc)
    
    
class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(cfg)
        self.enc_dec_attention = MultiHeadAttention(cfg)
        self.position_wise_feed_forward = PositionWiseFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg['d_model'])
        self.norm2 = nn.LayerNorm(cfg['d_model'])
        self.norm3 = nn.LayerNorm(cfg['d_model'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked multi-head attention
        attn_output = self.multi_head_attention(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Encoder-Decoder attention
        attn_output = self.enc_dec_attention(x, enc_output, enc_output, attn_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.position_wise_feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        self.cfg = cfg
        self.d_model = cfg['d_model']
        
        self.embed = nn.Embedding(cfg['num_tokens'], self.d_model, padding_idx=0)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg['num_layers'])])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        self.batch_size, self.seq_len = x.shape
        
        x = self.embed(x)
        x = x + self.get_position_encoding(self.seq_len, self.d_model).to(self.cfg['device'])
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x
    
    def get_position_encoding(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        pos_enc = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                theta = get_angle(pos, i, d_model)
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(theta)
                else:
                    pos_enc[pos, i] = np.cos(theta)
        return torch.FloatTensor(pos_enc)