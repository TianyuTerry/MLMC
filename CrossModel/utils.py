import multiprocessing
import pickle
import numpy as np
import sklearn
from typing import List, Tuple, Any
from transformers import *
import torch.nn as nn
from termcolor import colored
import os
import matplotlib.pyplot as plt

context_models = {
    'bert-base-uncased' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'bert-base-cased' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'bert-large-cased' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'bert-base-chinese' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'openai-gpt': {"model": OpenAIGPTModel, "tokenizer": OpenAIGPTTokenizer},
    'gpt2': {"model": GPT2Model, "tokenizer": GPT2Tokenizer},
    'ctrl': {"model": CTRLModel, "tokenizer": CTRLTokenizer},
    'transfo-xl-wt103': {"model": TransfoXLModel, "tokenizer": TransfoXLTokenizer},
    'xlnet-base-cased': {"model": XLNetModel, "tokenizer": XLNetTokenizer},
    'xlm-mlm-enfr-1024': {"model": XLMModel, "tokenizer": XLMTokenizer},
    'distilbert-base-cased': {"model": DistilBertModel, "tokenizer": DistilBertTokenizer},
    'roberta-base': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'roberta-large': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'xlm-roberta-base': {"model": XLMRobertaModel, "tokenizer": XLMRobertaTokenizer},
}

START_TAG = 'START'
STOP_TAG = 'STOP'
PAD_TAG = 'PAD'
sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
label2idx = {'O': 0, 'B': 1, 'I': 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}
idx2labels = ['O', 'B', 'I', START_TAG, STOP_TAG, PAD_TAG]
iobes_label2idx = {'O': 0, 'B': 1, 'I': 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5, 'E': 6, 'S': 7}
iobes_idx2labels = ['O', 'B', 'I', START_TAG, STOP_TAG, PAD_TAG, 'E', 'S']
semi_label2idx = {'O': 0, 'A': 1, START_TAG: 2, STOP_TAG: 3, PAD_TAG: 5}
semi_idx2labels = ['O', 'A', START_TAG, STOP_TAG, PAD_TAG]

B_PREF= "B"
I_PREF = "I"
S_PREF = "S"
E_PREF = "E"
O = "O"

def convert_bio_to_iobes(labels: List[str]) -> List[str]:
	"""
	Use IOBES tagging schema to replace the IOB tagging schema in the instance
	"""
	for pos in range(len(labels)):
		curr_entity = labels[pos]
		if pos == len(labels) - 1:
			if curr_entity.startswith(B_PREF):
				labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				labels[pos] = curr_entity.replace(I_PREF, E_PREF)
		else:
			next_entity = labels[pos + 1]
			if curr_entity.startswith(B_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(I_PREF, E_PREF)
	return labels

def convert_iobes_to_bio(labels: List[str]) -> List[str]:
    return [label.replace(E_PREF, I_PREF) if label.startswith(E_PREF) else label.replace(S_PREF, B_PREF) if label.startswith(S_PREF) else label for label in labels]

def convert_idx_to_iobes(label_idx: List[int]) -> List[str]:
    return [iobes_idx2labels[idx] for idx in label_idx]

def convert_bio_to_idx(labels: List[str]) -> List[int]:
    return [label2idx[label] for label in labels]

def init_kernel(layer: nn.Module):
    _, _, kernel_size, _ = layer.weight.data.size()
    nn.init.xavier_uniform_(layer.weight)
    layer.weight.data[:, :, kernel_size//2, kernel_size//2] = 500
    nn.init.uniform_(layer.bias)
    print(colored("Initializing 500 as kernel center weight for CNN predictor", color='yellow'))
    return layer


def get_huggingface_optimizer_and_scheduler(args, model: nn.Module,
                                            num_training_steps: int,
                                            weight_decay: float = 0.0,
                                            eps: float = 1e-8,
                                            warmup_step: int = 0):
    """
    Copying the optimizer code from HuggingFace.
    """
    print(colored(f"Using AdamW optimizer by HuggingFace with {args.lr} learning rate, "
                  f"eps: {eps}, weight decay: {weight_decay}, warmup_step: {warmup_step}, ", 'yellow'))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

class Metric():
    def __init__(self, args, predictions, goldens, review_lengths, reply_lengths, pred_bio_review, pred_bio_reply, golden_bio_review, golden_bio_reply):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.review_lengths = review_lengths
        self.reply_lengths = reply_lengths
        self.data_num = len(self.predictions) 
        self.golden_bio_review = golden_bio_review
        self.golden_bio_reply = golden_bio_reply
        if args.encoding_scheme == 'IOBES':
            self.pred_bio_review = [convert_bio_to_idx(convert_iobes_to_bio(convert_idx_to_iobes(pred_seq[::-1]))) for pred_seq in pred_bio_review]
            self.pred_bio_reply = [convert_bio_to_idx(convert_iobes_to_bio(convert_idx_to_iobes(pred_seq[::-1]))) for pred_seq in pred_bio_reply]
        else:
            self.pred_bio_review = [pred_seq[::-1] for pred_seq in pred_bio_review]
            self.pred_bio_reply = [pred_seq[::-1] for pred_seq in pred_bio_reply]

    def get_review_spans(self, biotags, review_lengths): # review
        spans = []
        start = -1
        for i in range(review_lengths):
            if biotags[i] == 1:
                start = i
                if i == review_lengths-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
                    start = -1 
            elif biotags[i] == 2:
                if i == review_lengths-1:
                    if start != -1:
                        spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, i])
                    start = -1
        return spans

    def get_reply_spans(self, biotags, reply_lengths): # rebuttal
        spans = []
        start = -1
        for i in range(reply_lengths):
            if biotags[i] == 1:
                start = i
                if i == reply_lengths-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
                    start = -1
            elif biotags[i] == 2:
                if i == reply_lengths-1:
                    if start != -1:
                        spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, i])
                    start = -1
        return spans

    def find_pair(self, tags, review_spans, reply_spans):
        pairs = []
        for al, ar in review_spans:
            for pl, pr in reply_spans:
                tag_num = [0] * self.args.class_num
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        tag_num[int(tags[i][j])] += 1
                        # tag_num[int(tags[j][i])] += 1
                if tag_num[self.args.class_num-1] < 1*(ar-al+1)*(pr-pl+1)*self.args.pair_threshold: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def score_review(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_review_spans = self.get_review_spans(self.golden_bio_review[i], self.review_lengths[i])
            for spans in golden_review_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_review_spans = self.get_review_spans(self.pred_bio_review[i], self.review_lengths[i])
            for spans in predicted_review_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        # precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        # recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # return precision, recall, f1
        return correct_num, len(predicted_set), len(golden_set)

    def score_reply(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_reply_spans = self.get_reply_spans(self.golden_bio_reply[i], self.reply_lengths[i])
            for spans in golden_reply_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_reply_spans = self.get_reply_spans(self.pred_bio_reply[i], self.reply_lengths[i])
            for spans in predicted_reply_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        # precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        # recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # return precision, recall, f1
        return correct_num, len(predicted_set), len(golden_set)

    def score_bio(self, review, reply):
        correct_num = review[0] + reply[0]
        pred_num = review[1] + reply[1]
        gold_num = review[2] + reply[2]
        precision = correct_num / pred_num * 100 if pred_num > 0 else 0
        recall = correct_num / gold_num * 100 if gold_num > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_pair(self):
        # self.all_labels: (batch_size, num_sents, num_sents)
        all_labels = [k for i in range(self.data_num) for j in self.goldens[i] for k in j]
        all_preds = [k for i in range(self.data_num) for j in self.predictions[i] for k in j]
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in range(len(all_labels)):
            if all_labels[i] != -1:
                if all_labels[i] == 1 and all_preds[i] == 1:
                    tp += 1
                elif all_labels[i] == 1 and all_preds[i] == 0:
                    fn += 1
                elif all_labels[i] == 0 and all_preds[i] == 1:
                    fp += 1
                elif all_labels[i] == 0 and all_preds[i] == 0:
                    tn += 1
        precision = 1.0 * tp / (tp + fp) * 100 if tp + fp != 0 else 0
        recall = 1.0 * tp / (tp + fn) * 100 if tp + fn != 0 else 0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_review_spans = self.get_review_spans(self.golden_bio_review[i], self.review_lengths[i])
            golden_reply_spans = self.get_reply_spans(self.golden_bio_reply[i], self.reply_lengths[i])
            golden_tuples = self.find_pair(self.goldens[i], golden_review_spans, golden_reply_spans)
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_review_spans = self.get_review_spans(self.pred_bio_review[i], self.review_lengths[i])
            predicted_reply_spans = self.get_reply_spans(self.pred_bio_reply[i], self.reply_lengths[i])
            predicted_tuples = self.find_pair(self.predictions[i], predicted_review_spans, predicted_reply_spans)
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) * 100 if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) * 100 if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1


class Writer():
    """
    output test dataset results to file
    """
    def __init__(self, args, predictions, goldens, review_lengths, reply_lengths, pred_bio_review, pred_bio_reply, golden_bio_review, golden_bio_reply):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.review_lengths = review_lengths
        self.reply_lengths = reply_lengths
        self.data_num = len(self.predictions)
        self.golden_bio_review = golden_bio_review
        self.golden_bio_reply = golden_bio_reply
        self.pred_bio_review = [pred_seq[::-1] for pred_seq in pred_bio_review]
        self.pred_bio_reply = [pred_seq[::-1] for pred_seq in pred_bio_reply]
        self.output_dir = os.path.join(args.output_dir, args.model_dir.strip('/'), args.model_dir.strip('/') + '.txt')
    
    def get_spans(self, biotags, lengths):
        spans = []
        start = -1
        for i in range(lengths):
            if biotags[i] == 1:
                start = i
                if i == lengths-1:
                    spans.append([start, i])
                elif biotags[i+1] != 2:
                    spans.append([start, start])
                    start = -1
            elif biotags[i] == 2:
                if i == lengths-1:
                    if start != -1:
                        spans.append([start, i])
                elif biotags[i+1] != 2 and start != -1:
                    spans.append([start, i])
                    start = -1
        return spans
    
    def find_pair(self, tags, review_spans, reply_spans):
        pairs = []
        for al, ar in review_spans:
            for pl, pr in reply_spans:
                tag_num = [0] * self.args.class_num
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        tag_num[int(tags[i][j])] += 1
                if tag_num[self.args.class_num-1] < 1*(ar-al+1)*(pr-pl+1)*self.args.pair_threshold: continue
                pairs.append([al, ar, pl, pr])
        return pairs
    
    def output_results(self):
        with open(self.output_dir, 'w') as f:
            f.write('\t'.join(['review_golden', 'review_pred', 'reply_golden', 'reply_pred', 'pair_golden', 'pair_pred', 'pair_golden_len']) + '\n')
            for i in range(self.data_num):
                golden_review_spans = self.get_spans(self.golden_bio_review[i], self.review_lengths[i])
                review_golden = '|'.join(map(lambda span: '-'.join(map(str, span)), golden_review_spans))

                predicted_review_spans = self.get_spans(self.pred_bio_review[i], self.review_lengths[i])
                review_pred = '|'.join(map(lambda span: '-'.join(map(str, span)), predicted_review_spans))
                
                golden_reply_spans = self.get_spans(self.golden_bio_reply[i], self.reply_lengths[i])
                reply_golden = '|'.join(map(lambda span: '-'.join(map(str, span)), golden_reply_spans))

                predicted_reply_spans = self.get_spans(self.pred_bio_reply[i], self.reply_lengths[i])
                reply_pred = '|'.join(map(lambda span: '-'.join(map(str, span)), predicted_reply_spans))
                
                golden_pairs = self.find_pair(self.goldens[i], golden_review_spans, golden_reply_spans)
                pair_golden = '|'.join(map(lambda pair: '-'.join(map(str, pair)), golden_pairs))

                predicted_pairs = self.find_pair(self.predictions[i], predicted_review_spans, predicted_reply_spans)
                pair_pred = '|'.join(map(lambda pair: '-'.join(map(str, pair)), predicted_pairs))

                f.write('\t'.join([review_golden, review_pred, reply_golden, reply_pred, pair_golden, pair_pred, str(len(golden_pairs))]) + '\n')
        
       
def plot_attention_weights(attention, review, reply, output_dir):
    
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax_r = ax.secondary_yaxis('right')
    ax_b = ax.secondary_xaxis('bottom')
    ax.matshow(attention, cmap='Blues', origin='upper')
    """
    dynamic font size adjustment according to input size
    """
    fontdict_digit = {'fontsize': 35*15/max(len(review), len(reply))}
    fontdict_char = {'fontsize': 35*15/max(len(review), len(reply))}

    ax.set_xticks(range(len(reply)))
    ax.set_yticks(range(len(review)))
    ax.set_xticklabels(['${}$'.format(str(i)) for i in range(1, len(reply)+1)], fontdict=fontdict_digit)
    ax.set_yticklabels(['${}$'.format(str(i)) for i in range(1, len(review)+1)], fontdict=fontdict_digit)
    
    ax_r.set_yticks(range(len(review)))
    ax_r.tick_params(axis='y', direction='out')
    ax_r.set_yticklabels(['${}$'.format(label) for label in review], fontdict=fontdict_char)

    plt.rcParams["font.family"] = "Times New Roman"
    ax_b.set_xticks(range(len(reply)))
    ax_b.tick_params(axis='x', direction='out')
    ax_b.set_xticklabels(['${}$'.format(label) for label in reply], fontdict=fontdict_char)

    fig.tight_layout()
    plt.plot()
    plt.savefig(output_dir)
    plt.clf()


def plot_attn_loss(maxnumepochs, model, train_losses, val_losses, test_losses, output_dir):
    ls_x = range(1, maxnumepochs+1)
    plt.title('loss for ' + model)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(ls_x, train_losses, color="red", label="train attn loss")
    plt.plot(ls_x, val_losses, color="green", label="validation attn loss")
    plt.plot(ls_x, test_losses, color="blue", label="test attn loss")
    plt.legend()
    plt.savefig(output_dir)
    plt.clf()