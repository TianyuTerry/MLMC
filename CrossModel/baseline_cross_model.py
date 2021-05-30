import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from termcolor import colored

from module.rnn import BiLSTMEncoder, GRU2dLayer, BGRU2dLayer
from module.inferencer import LinearCRF
from module.embedder import Embedder, TokenEmbedder
from module.attention import CrossAttentionLayer, CrossAttentionCosineSimilarityLayer

from data import START_TAG, STOP_TAG, PAD_TAG, label2idx, idx2labels, iobes_idx2labels, iobes_label2idx
from typing import Dict, List, Tuple, Any
from utils import init_kernel

class JointModule(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, bidirectional, layer_norm, lstm_share_param, attention='tanh', output_dim=None, dropout=0.5):
        super(JointModule, self).__init__()
        self.lstm_share_param = lstm_share_param
        if lstm_share_param:
            self.lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)
        else:
            self.review_lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)
            self.reply_lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)
        if bidirectional:
            self.table_processer = BGRU2dLayer(hidden_dim*2, hidden_dim, layer_norm)
        else:
            self.table_processer = GRU2dLayer(hidden_dim*2, hidden_dim*2, layer_norm)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim*2)
        self.table_transform = nn.Linear(hidden_dim*2, hidden_dim*2)
        if attention == 'tanh':
            self.attn_layer = CrossAttentionLayer(hidden_dim*2, hidden_dim, output_dim)
        elif attention == 'cosine_similarity':
            self.attn_layer = CrossAttentionCosineSimilarityLayer(hidden_dim*2, hidden_dim, output_dim)
    
    def forward(self, review_input, reply_input, table_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens):
        """
        Encoding the input with RNNs
        param:
        review_input: (batch_size, num_review_sents, input_dim)
        reply_input: (batch_size, num_reply_sents, input_dim)
        table_input: (batch_size, num_review_sents, num_reply_sents, hidden_dim*2)
        review_seq_lens: (batch_size, )
        reply_seq_lens: (batch_size, )
        review_input_mask: (batch_size, num_review_sents)
        reply_input_mask: (batch_size, num_reply_sents)
        return: 
        review_feature_out: (batch_size, num_review_sents, hidden_dim)
        reply_feature_out: (batch_size, num_reply_sents, hidden_dim)
        table_feature_out: (batch_size, num_review_sents, num_reply_sents, hidden_dim)
        """
        if self.lstm_share_param:
            review_lstm_embedding = self.lstm_encoder(review_input, review_seq_lens)
            reply_lstm_embedding = self.lstm_encoder(reply_input, reply_seq_lens)
        else:
            review_lstm_embedding = self.review_lstm_encoder(review_input, review_seq_lens)
            reply_lstm_embedding = self.reply_lstm_encoder(reply_input, reply_seq_lens)
        _, num_review_sents, _ = review_lstm_embedding.size()
        _, num_reply_sents, _ = reply_lstm_embedding.size()
        expanded_review_embedding = review_lstm_embedding.unsqueeze(2).expand([-1, -1, num_reply_sents, -1])
        expanded_reply_embedding = reply_lstm_embedding.unsqueeze(1).expand([-1, num_review_sents, -1, -1])
        cross_feature = torch.cat((expanded_review_embedding, expanded_reply_embedding), dim=3)
        cross_feature = self.table_transform(cross_feature)
        cross_feature = self.ln(self.dropout(cross_feature) + table_input)
        cross_mask = reply_input_mask.unsqueeze(1) * review_input_mask.unsqueeze(2)
        grid_feature, _ = self.table_processer(cross_feature, cross_mask)
        attn_review_feature, attn_reply_feature, attn_review, attn_reply = self.attn_layer(grid_feature, review_lstm_embedding, reply_lstm_embedding, review_input_mask, reply_input_mask)
        return grid_feature, attn_review_feature, attn_reply_feature, attn_review, attn_reply

class BaselineCrossModel(nn.Module):
    
    def __init__(self, args, bertModel):
        super(BaselineCrossModel, self).__init__()
        if args.token_embedding:
            self.embedder = TokenEmbedder(bertModel, args)
        else:
            self.embedder = Embedder(bertModel)
        self.encoder = nn.ModuleList([JointModule(args.bert_feature_dim, args.hidden_dim, args.bidirectional, args.layer_norm, args.lstm_share_param) if i == 0 \
                                      else JointModule(args.hidden_dim, args.hidden_dim, args.bidirectional, args.layer_norm, args.lstm_share_param) for i in range(args.iteration)])
        self.share_crf_param = args.share_crf_param
        self.cnn_classifier = args.cnn_classifier
        if args.encoding_scheme == 'IOBES':
            crf_label2idx, crf_idx2labels, iobes = iobes_label2idx, iobes_idx2labels, True
        else:
            crf_label2idx, crf_idx2labels, iobes = label2idx, idx2labels, False
        if args.share_crf_param:
            self.review_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
            self.reply_crf = self.review_crf
        else:
            self.review_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
            self.reply_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
        if args.cnn_classifier:
            self.reduce_dim = nn.Sequential(nn.Linear(args.hidden_dim*2, 100), nn.ReLU())
            classifier = [nn.Conv2d(100, 50, args.kernel_size, padding=args.kernel_size//2), nn.ReLU(), nn.Conv2d(50, args.class_num, args.kernel_size, padding=args.kernel_size//2)]
        else:
            classifier = [nn.Linear(args.hidden_dim*2, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, args.class_num)]
        self.hidden2tag = nn.Sequential(*classifier)
        self.hidden2biotag = nn.Linear(args.hidden_dim, len(crf_label2idx))
        self.initial_table_input = torch.zeros((1, 1, 1, args.hidden_dim*2)).to(args.device)
        self.args = args
        info = f"""[model info] model: BaselineCrossModel
                BERT model: {args.bert_model_path} 
                iterations: {args.iteration}
                pair_loss weight: {args.pair_weight}
                bert_feature_dim: {args.bert_feature_dim} 
                hidden_dim: {args.hidden_dim}"""
        print(colored(info, 'yellow'))
    
    def forward(self, review_embedder_input, reply_embedder_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens, review_bio_tags, reply_bio_tags):
        
        review_feature = self.embedder(*review_embedder_input)
        reply_feature = self.embedder(*reply_embedder_input)
        batch_size, review_num_sents, _ = review_feature.size()
        _, reply_num_sents, _ = reply_feature.size()
        grid_feature = self.initial_table_input.expand(batch_size, review_num_sents, reply_num_sents, -1)
        # attn_sum = torch.zeros((batch_size, review_num_sents, reply_num_sents)).to(self.args.device)
        dev_num = review_feature.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        attn_sum = torch.zeros((batch_size, review_num_sents, reply_num_sents), device=curr_dev)
        for encoder_module in self.encoder:
            grid_feature, review_feature, reply_feature, attn_review, attn_reply = encoder_module(review_feature, reply_feature, grid_feature, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens)
            attn_sum = self.args.ema*attn_sum + attn_review + attn_reply.permute(0, 2, 1)
        review_crf_input = self.hidden2biotag(review_feature)
        reply_crf_input = self.hidden2biotag(reply_feature)
        review_crf_loss = self.review_crf(review_crf_input, review_seq_lens, review_bio_tags, review_input_mask)
        reply_crf_loss = self.reply_crf(reply_crf_input, reply_seq_lens, reply_bio_tags, reply_input_mask)
        if self.cnn_classifier:
            grid_feature = self.reduce_dim(grid_feature)
            pair_output = self.hidden2tag(grid_feature.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        else:
            pair_output = self.hidden2tag(grid_feature)

        return pair_output, review_crf_loss + reply_crf_loss, attn_sum
    
    def decode(self, review_embedder_input, reply_embedder_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens):
        
        review_feature = self.embedder(*review_embedder_input)
        reply_feature = self.embedder(*reply_embedder_input)
        batch_size, review_num_sents, _ = review_feature.size()
        _, reply_num_sents, _ = reply_feature.size()
        grid_feature = self.initial_table_input.expand(batch_size, review_num_sents, reply_num_sents, -1)
        # attn_sum = torch.zeros((batch_size, review_num_sents, reply_num_sents)).to(self.args.device)
        dev_num = review_feature.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        attn_sum = torch.zeros((batch_size, review_num_sents, reply_num_sents), device=curr_dev)
        for encoder_module in self.encoder:
            grid_feature, review_feature, reply_feature, attn_review, attn_reply = encoder_module(review_feature, reply_feature, grid_feature, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens)
            attn_sum = self.args.ema*attn_sum + attn_review + attn_reply.permute(0, 2, 1)
        review_crf_input = self.hidden2biotag(review_feature)
        reply_crf_input = self.hidden2biotag(reply_feature)
        _, review_decode_idx = self.review_crf.decode(review_crf_input, review_seq_lens)
        _, reply_decode_idx = self.reply_crf.decode(reply_crf_input, reply_seq_lens)
        if self.cnn_classifier:
            grid_feature = self.reduce_dim(grid_feature)
            pair_output = self.hidden2tag(grid_feature.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        else:
            pair_output = self.hidden2tag(grid_feature)

        return pair_output, review_decode_idx, reply_decode_idx, attn_sum