import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.jit as jit
from typing import Optional

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder
    """
    def __init__(self, input_dim, hidden_dim, drop_lstm=0.5, num_lstm_layers=1):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)

    def forward(self, sent_rep: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        param:
        sent_rep: (batch_size, num_sents, emb_size)
        seq_lens: (batch_size, )
        return: 
        feature_out: (batch_size, num_sents, hidden_dim)
        """
        sorted_seq_len, permIdx = seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = sent_rep[permIdx]

        packed_sents = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        lstm_out, _ = self.lstm(packed_sents, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        feature_out = self.drop_lstm(lstm_out)

        return feature_out[recover_idx]


class GRU2dCell(nn.Module):
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.Wi = nn.Linear(self.input_dim, self.state_dim*4)
        self.Ws = nn.Linear(self.state_dim*2, self.state_dim*4)

    def forward(self, x, s_prev0, s_prev1):
       
        s = torch.cat([s_prev0, s_prev1], -1)
        igates = self.Wi(x)
        sgates = self.Ws(s)
        gates = igates + sgates

        # r_inv actual represents (1-r)
        r_inv, i, n, l = gates.chunk(4, 1)
        s_n = sgates[:, self.state_dim*2:self.state_dim*3]
        
        l = l.sigmoid()
        r_inv = r_inv.sigmoid()
        i = i.sigmoid()
        n = (n - r_inv*s_n).tanh() # <==> (i_n + r * s_n)
        
        h = n + i * (l*s_prev0 + (1.-l)*s_prev1 - n)

        return h

class GRU2dLayer(nn.Module):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, emb_dim, hidden_dim, layer_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        if layer_norm:
            self.cell = LNGRU2dCell(self.emb_dim, self.hidden_dim, dropout=0.0)
        else:
            self.cell = GRU2dCell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    def forward(self, x: torch.Tensor, masks: torch.Tensor, states: Optional[torch.Tensor] = None):
        
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = x.flip(1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float()
        masks = masks.flip(1)
        
        if states is None:
            states = torch.zeros(T0+1, T1+1, B, H, device=x.device) # (T0+1, T1+1*, B, H)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E)
        
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev0 = s_current[-diag_len:].view(new_batch_size, H)
            s_prev1 = s_current[:diag_len].view(new_batch_size, H)
            
            s_next = self.cell(x_current, s_prev0, s_prev1)
            
            to_save = s_next.view(diag_len, B, H)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            
            if T0 < T1 and len(to_save) == len(s_next) - 1:
                s_next[1:] = to_save
            elif T0 > T1 and len(to_save) == len(s_next) - 1:
                s_next[:-1] = to_save
            else:
                s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        states_s = states_s.flip(2)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)

class BGRU2dLayer(nn.Module):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, emb_dim, hidden_dim, layer_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        
        if layer_norm:
            self.cellf = LNGRU2dCell(self.emb_dim, self.hidden_dim, dropout=0.0)
            self.cellb = LNGRU2dCell(self.emb_dim, self.hidden_dim, dropout=0.0)
        else:
            self.cellf = GRU2dCell(self.emb_dim, self.hidden_dim, dropout=0.0)
            self.cellb = GRU2dCell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    def forward(self, 
                x: torch.Tensor, 
                masks: torch.Tensor,
                states: Optional[torch.Tensor] = None):
        
        assert states is None
        
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x.flip(1), x.flip(0)], -1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks.flip(1), masks.flip(0)], -1)
        
        states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, H)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*2)
            x_current_f, x_current_b = x_current.chunk(2, -1)
            
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_f0, s_prev_b0 = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f1, s_prev_b1 = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            
            s_next_f = self.cellf(
                x_current_f, s_prev_f0, s_prev_f1)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1)
            
            to_save = torch.cat([s_next_f, s_next_b], -1).view(diag_len, B, H*2)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            
            if T0 < T1 and len(to_save) == len(s_next) - 1:
                s_next[1:] = to_save
            elif T0 > T1 and len(to_save) == len(s_next) - 1:
                s_next[:-1] = to_save
            else:
                s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1)], -1)
        
        return states_s, states

class LNGRU2dCell(nn.Module):
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.Wi = nn.Linear(self.input_dim, self.state_dim * 4, bias=None)
        self.Ws = nn.Linear(self.state_dim * 2, self.state_dim * 4, bias=None)
        self.LNi = nn.LayerNorm(self.state_dim * 4)
        self.LNs = nn.LayerNorm(self.state_dim * 4)
        self.LNh = nn.LayerNorm(self.state_dim)
        self.dropout_layer = nn.Dropout(dropout, inplace=True)

    def forward(self, x, s_prev0, s_prev1):
        
        s = torch.cat([s_prev0, s_prev1], -1)
        igates = self.dropout_layer(self.LNi(self.Wi(x)))
        sgates = self.dropout_layer(self.LNs(self.Ws(s)))
        gates = igates + sgates

        # r_inv actual represents (1-r)
        r_inv, i, n, l = gates.chunk(4, 1)
        s_n = sgates[:, self.state_dim*2:self.state_dim*3]
        
        l = l.sigmoid()
        r_inv = r_inv.sigmoid()
        i = i.sigmoid()
        n = (n - r_inv*s_n).tanh() # <==> (i_n + r * s_n)
        
        h = n + i * (l*s_prev0 + (1.-l)*s_prev1 - n)
        
        h = self.dropout_layer(self.LNh(h))

        return h