import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):

    def __init__(self, table_dim, input_dim, output_dim=None, dropout=0.5):
        super(AttentionLayer, self).__init__()
        self.V = nn.Linear(table_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)
        if output_dim:
            self.hidden2tag = nn.Linear(input_dim, output_dim)
        else:
            self.hidden2tag = None
    
    def forward(self, table, sent_rep, input_mask):
        attn = F.softmax(torch.tanh(self.V(table)).squeeze(-1), dim=-1)
        attn = attn * input_mask.unsqueeze(1)
        attn_sum = torch.sum(attn, dim=-1, keepdim=True)
        attn /= attn_sum
        sent_rep_attn = torch.bmm(attn, sent_rep)
        sent_rep_new = self.ln(self.dropout(sent_rep_attn) + sent_rep)
        if self.hidden2tag:
            return self.hidden2tag(sent_rep_new), attn
        return sent_rep_new, attn
        

class AttentionCosineSimilarityLayer(nn.Module):

    def __init__(self, table_dim, input_dim, output_dim, dropout=0.5):
        super(AttentionCosineSimilarityLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)
        if output_dim:
            self.hidden2tag = nn.Linear(input_dim, output_dim)
        else:
            self.hidden2tag = None
    
    def forward(self, table, sent_rep, input_mask):
        review_table, reply_table = torch.chunk(table, 2, dim=-1)
        cosine_similarity = F.cosine_similarity(review_table, reply_table, dim=-1)

        attn = F.softmax(cosine_similarity, dim=-1)
        attn = attn * input_mask.unsqueeze(1)
        attn_sum = torch.sum(attn, dim=-1, keepdim=True)
        attn /= attn_sum
        sent_rep_attn = torch.bmm(attn, sent_rep)
        sent_rep_new = self.ln(self.dropout(sent_rep_attn) + sent_rep)
        if self.hidden2tag:
            return self.hidden2tag(sent_rep_new), attn
        return sent_rep_new, attn

class CrossAttentionLayer(nn.Module):

    def __init__(self, table_dim, input_dim, output_dim=None, dropout=0.5):
        super(CrossAttentionLayer, self).__init__()
        self.V_1 = nn.Linear(table_dim, 1)
        self.V_2 = nn.Linear(table_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(input_dim)
        self.ln_2 = nn.LayerNorm(input_dim)
        if output_dim:
            self.hidden2tag = nn.Linear(input_dim, output_dim)
        else:
            self.hidden2tag = None
    
    def forward(self, table, review_sent_rep, reply_sent_rep, review_input_mask, reply_input_mask):
        """
        Cross Attention
        param:
        table: (batch_size, num_review_sents, num_reply_sents, table_dim)
        review_sent_rep: (batch_size, num_review_sents, input_dim)
        reply_sent_rep: (batch_size, num_reply_sents, input_dim)
        review_input_mask: (batch_size, num_review_sents)
        reply_input_mask: (batch_size, num_reply_sents)
        return: 
        review_feature_out: (batch_size, num_review_sents, hidden_dim)
        reply_feature_out: (batch_size, num_reply_sents, hidden_dim)
        table_feature_out: (batch_size, num_review_sents, num_reply_sents, hidden_dim)
        """
        attn_1 = F.softmax(torch.tanh(self.V_1(table)).squeeze(-1), dim=-1)
        attn_1 = attn_1 * reply_input_mask.unsqueeze(1)
        attn_sum_1 = torch.sum(attn_1, dim=-1, keepdim=True)
        attn_1 /= attn_sum_1
        review_sent_rep_attn = torch.bmm(attn_1, reply_sent_rep)
        review_sent_rep_new = self.ln_1(self.dropout(review_sent_rep_attn) + review_sent_rep)
        
        attn_2 = F.softmax(torch.tanh(self.V_2(table)).squeeze(-1).permute(0, 2, 1), dim=-1)
        attn_2 = attn_2 * review_input_mask.unsqueeze(1)
        attn_sum_2 = torch.sum(attn_2, dim=-1, keepdim=True)
        attn_2 /= attn_sum_2
        reply_sent_rep_attn = torch.bmm(attn_2, review_sent_rep)
        reply_sent_rep_new = self.ln_2(self.dropout(reply_sent_rep_attn) + reply_sent_rep)
        
        if self.hidden2tag:
            return self.hidden2tag(review_sent_rep_new), self.hidden2tag(reply_sent_rep_new), attn_1, attn_2
        return review_sent_rep_new, reply_sent_rep_new, attn_1, attn_2


class CrossAttentionCosineSimilarityLayer(nn.Module):

    def __init__(self, table_dim, input_dim, output_dim=None, dropout=0.5):
        super(CrossAttentionCosineSimilarityLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(input_dim)
        self.ln_2 = nn.LayerNorm(input_dim)
        if output_dim:
            self.hidden2tag = nn.Linear(input_dim, output_dim)
        else:
            self.hidden2tag = None
    
    def forward(self, table, review_sent_rep, reply_sent_rep, review_input_mask, reply_input_mask):
        """
        Cross Attention
        param:
        table: (batch_size, num_review_sents, num_reply_sents, table_dim)
        review_sent_rep: (batch_size, num_review_sents, input_dim)
        reply_sent_rep: (batch_size, num_reply_sents, input_dim)
        review_input_mask: (batch_size, num_review_sents)
        reply_input_mask: (batch_size, num_reply_sents)
        return: 
        review_feature_out: (batch_size, num_review_sents, hidden_dim)
        reply_feature_out: (batch_size, num_reply_sents, hidden_dim)
        table_feature_out: (batch_size, num_review_sents, num_reply_sents, hidden_dim)
        """
        review_table, reply_table = torch.chunk(table, 2, dim=-1)
        cosine_similarity = F.cosine_similarity(review_table, reply_table, dim=-1)
        
        attn_1 = F.softmax(cosine_similarity, dim=-1)
        attn_1 = attn_1 * reply_input_mask.unsqueeze(1)
        attn_sum_1 = torch.sum(attn_1, dim=-1, keepdim=True)
        attn_1 /= attn_sum_1
        review_sent_rep_attn = torch.bmm(attn_1, reply_sent_rep)
        review_sent_rep_new = self.ln_1(self.dropout(review_sent_rep_attn) + review_sent_rep)
        
        attn_2 = F.softmax(cosine_similarity.permute(0, 2, 1), dim=-1)
        attn_2 = attn_2 * review_input_mask.unsqueeze(1)
        attn_sum_2 = torch.sum(attn_2, dim=-1, keepdim=True)
        attn_2 /= attn_sum_2
        reply_sent_rep_attn = torch.bmm(attn_2, review_sent_rep)
        reply_sent_rep_new = self.ln_2(self.dropout(reply_sent_rep_attn) + reply_sent_rep)
        
        if self.hidden2tag:
            return self.hidden2tag(review_sent_rep_new), self.hidden2tag(reply_sent_rep_new), attn_1, attn_2
        return review_sent_rep_new, reply_sent_rep_new, attn_1, attn_2