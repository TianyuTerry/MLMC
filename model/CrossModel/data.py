import math
import torch
import numpy as np
from transformers import *
from utils import context_models, START_TAG, STOP_TAG, PAD_TAG, sentiment2id, label2idx, idx2labels, semi_label2idx, semi_idx2labels, O
from utils import iobes_label2idx, iobes_idx2labels, convert_bio_to_iobes
import itertools

def get_spans(tags):
    """
    for spans
    """
    tags = tags.strip().split('<tag>')
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans

class Instance(object):
    
    def __init__(self, tokenizer, sentence_pack, args, is_train):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.last_review = sentence_pack['split_idx']
        self.sents = self.sentence.strip().split(' <sentsep> ')
        self.review = self.sents[:self.last_review+1]
        self.reply = self.sents[self.last_review+1:]
        self.sen_length = len(self.sents)
        self.review_length = self.last_review + 1
        self.reply_length = self.sen_length - self.last_review - 1
        self.review_bert_tokens = []
        self.reply_bert_tokens = []
        self.review_num_tokens = []
        self.reply_num_tokens = []
        for i, sent in enumerate(self.review):
            word_tokens = tokenizer.tokenize(" " + sent)
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + word_tokens + [tokenizer.sep_token])
            self.review_bert_tokens.append(input_ids)
            self.review_num_tokens.append(min(len(word_tokens), args.max_bert_token-1))
        for i, sent in enumerate(self.reply):
            word_tokens = tokenizer.tokenize(" " + sent)
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + word_tokens + [tokenizer.sep_token])
            self.reply_bert_tokens.append(input_ids)
            self.reply_num_tokens.append(min(len(word_tokens), args.max_bert_token-1))
        self.length = len(self.sents)
        if is_train:
            self.tags = torch.full((self.review_length, self.reply_length), -1, dtype=torch.long)
        else:
            self.tags = torch.zeros(self.review_length, self.reply_length).long()
        
        review_bio_list = [O] * self.review_length
        reply_bio_list = [O] * self.reply_length

        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            for l, r in aspect_span:
                for i in range(l, r+1):
                    if i == l:
                        review_bio_list[i] = 'B'
                    else:
                        review_bio_list[i] = 'I'

            for l, r in opinion_span:
                for i in range(l, r+1):
                    if i == l:
                        reply_bio_list[i-self.review_length] = 'B'
                    else:
                        reply_bio_list[i-self.review_length] = 'I'

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            self.tags[i][j-self.review_length] = 1
        
        if args.encoding_scheme == 'BIO' or not is_train:
            review_bio_list = [label2idx[label] for label in review_bio_list]
            reply_bio_list = [label2idx[label] for label in reply_bio_list]
            self.review_bio = torch.LongTensor(review_bio_list)
            self.reply_bio = torch.LongTensor(reply_bio_list)
        elif args.encoding_scheme == 'IOBES' and is_train:
            review_bio_list = [iobes_label2idx[label] for label in convert_bio_to_iobes(review_bio_list)]
            reply_bio_list = [iobes_label2idx[label] for label in convert_bio_to_iobes(reply_bio_list)]
            self.review_bio = torch.LongTensor(review_bio_list)
            self.reply_bio = torch.LongTensor(reply_bio_list)

def load_data_instances(sentence_packs, args, is_train):
    instances = list()
    tokenizer = context_models[args.bert_tokenizer_path]['tokenizer'].from_pretrained(args.bert_tokenizer_path)
    if args.num_instances != -1:
        for sentence_pack in sentence_packs[:args.num_instances]:
            instances.append(Instance(tokenizer, sentence_pack, args, is_train))
    else:
        for sentence_pack in sentence_packs:
            instances.append(Instance(tokenizer, sentence_pack, args, is_train))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)
        self.max_bert_token = args.max_bert_token

    def __len__(self):
        return len(self.instances)

    def get_batch(self, index):
        sentence_ids = []
        reviews = []
        replies = []
        sens_lens = []
        lengths = []
        review_lengths = []
        reply_lengths = [] 

        batch_size = min((index + 1) * self.args.batch_size, len(self.instances)) - index * self.args.batch_size
        max_review_num_sents = max([self.instances[i].review_length for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))])
        max_reply_num_sents = max([self.instances[i].reply_length for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))])
        max_review_sent_length = min(max([max(map(len, self.instances[i].review_bert_tokens)) for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))]), self.max_bert_token)
        max_reply_sent_length = min(max([max(map(len, self.instances[i].reply_bert_tokens)) for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))]), self.max_bert_token)

        review_bert_tokens = torch.zeros(batch_size, max_review_num_sents, max_review_sent_length, dtype=torch.long)
        reply_bert_tokens = torch.zeros(batch_size, max_reply_num_sents, max_reply_sent_length, dtype=torch.long)
        review_attn_masks = torch.zeros(batch_size, max_review_num_sents, max_review_sent_length, dtype=torch.long)
        reply_attn_masks = torch.zeros(batch_size, max_reply_num_sents, max_reply_sent_length, dtype=torch.long)
        review_masks = torch.zeros(batch_size, max_review_num_sents, dtype=torch.long)
        reply_masks = torch.zeros(batch_size, max_reply_num_sents, dtype=torch.long)
        tags = -torch.ones(batch_size, max_review_num_sents, max_reply_num_sents).long()
        review_biotags = torch.full((batch_size, max_review_num_sents), label2idx[PAD_TAG]).long()
        reply_biotags = torch.full((batch_size, max_reply_num_sents), label2idx[PAD_TAG]).long()
        review_num_tokens = torch.ones(batch_size, max_review_num_sents).long()
        reply_num_tokens = torch.ones(batch_size, max_reply_num_sents).long()

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            reviews.append(self.instances[i].review)
            replies.append(self.instances[i].reply)
            sens_lens.append(self.instances[i].sen_length)
            lengths.append(self.instances[i].length)
            review_lengths.append(self.instances[i].review_length)
            reply_lengths.append(self.instances[i].reply_length)
            review_masks[i-index * self.args.batch_size, :self.instances[i].review_length] = 1
            reply_masks[i-index * self.args.batch_size, :self.instances[i].reply_length] = 1
            tags[i-index * self.args.batch_size, :self.instances[i].review_length, :self.instances[i].reply_length] = self.instances[i].tags
            review_biotags[i-index * self.args.batch_size, :self.instances[i].review_length] = self.instances[i].review_bio
            reply_biotags[i-index * self.args.batch_size, :self.instances[i].reply_length] = self.instances[i].reply_bio
            review_num_tokens[i-index * self.args.batch_size, :self.instances[i].review_length] = torch.LongTensor(self.instances[i].review_num_tokens)
            reply_num_tokens[i-index * self.args.batch_size, :self.instances[i].reply_length] = torch.LongTensor(self.instances[i].reply_num_tokens)

            for j in range(self.instances[i].review_length):
                length_filled = min(self.max_bert_token, len(self.instances[i].review_bert_tokens[j]))
                review_bert_tokens[i-index * self.args.batch_size, j, :length_filled] = \
                    torch.LongTensor(self.instances[i].review_bert_tokens[j][:length_filled])
                review_attn_masks[i-index * self.args.batch_size, j, :length_filled] = 1
            for k in range(self.instances[i].reply_length):
                length_filled = min(self.max_bert_token, len(self.instances[i].reply_bert_tokens[k]))
                reply_bert_tokens[i-index * self.args.batch_size, k, :length_filled] = \
                    torch.LongTensor(self.instances[i].reply_bert_tokens[k][:length_filled])
                reply_attn_masks[i-index * self.args.batch_size, k, :length_filled] = 1

        review_bert_tokens = review_bert_tokens.to(self.args.device)
        reply_bert_tokens = reply_bert_tokens.to(self.args.device)
        review_attn_masks = review_attn_masks.to(self.args.device)
        reply_attn_masks = reply_attn_masks.to(self.args.device)
        tags = tags.to(self.args.device)
        review_masks = review_masks.to(self.args.device)
        reply_masks = reply_masks.to(self.args.device)
        review_biotags = review_biotags.to(self.args.device)
        reply_biotags = reply_biotags.to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        review_lengths = torch.tensor(review_lengths).to(self.args.device)
        reply_lengths = torch.tensor(reply_lengths).to(self.args.device)
        review_num_tokens = review_num_tokens.to(self.args.device)
        reply_num_tokens = reply_num_tokens.to(self.args.device)

        if self.args.token_embedding:
            return sentence_ids, reviews, replies, (review_bert_tokens, review_attn_masks, review_num_tokens), (reply_bert_tokens, reply_attn_masks, reply_num_tokens), lengths, sens_lens, tags, review_biotags, reply_biotags, review_masks, reply_masks, reply_lengths, review_lengths
        else:
            return sentence_ids, reviews, replies, (review_bert_tokens, review_attn_masks), (reply_bert_tokens, reply_attn_masks), lengths, sens_lens, tags, review_biotags, reply_biotags, review_masks, reply_masks, reply_lengths, review_lengths
