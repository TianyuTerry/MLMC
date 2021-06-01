import json, os
import random
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import trange

from data import load_data_instances, DataIterator, idx2labels
from cross_model import CrossModel
from utils import get_huggingface_optimizer_and_scheduler, context_models, Metric, Writer, plot_attention_weights, plot_attn_loss
import math
from termcolor import colored

def train(args):

    random.seed(args.random_seed)
    
    if not args.test_code:
        train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
        random.shuffle(train_sentence_packs)
        dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))
        test_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    else:
        train_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
        dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
        test_sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))

    train_batch_count = math.ceil(len(train_sentence_packs)/args.batch_size)

    instances_train = load_data_instances(train_sentence_packs, args, is_train=True)
    instances_dev = load_data_instances(dev_sentence_packs, args, is_train=False)
    instances_test = load_data_instances(test_sentence_packs, args, is_train=False)
    del train_sentence_packs
    del dev_sentence_packs
    del test_sentence_packs
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)
    testset = DataIterator(instances_test, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.model_name)):
        os.makedirs(os.path.join(args.output_dir, args.model_name))
    bertModel = context_models[args.bert_model_path]['model'].from_pretrained(args.bert_model_path, return_dict=False)

    model = CrossModel(args, bertModel).to(args.device)
    
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(args, model, num_training_steps=train_batch_count * args.epochs,
                                                                    weight_decay=0.0, eps = 1e-8, warmup_step=0)

    best_joint_f1 = -1
    best_joint_epoch = 0
    best_joint_precision = 0
    joint_precision = 0
    best_joint_recall = 0
    attn_losses_all = []
    attn_losses_dev = []
    attn_losses_test = []
    
    for epoch in range(1, args.epochs+1):
        
        model.zero_grad()
        model.train()
        print('Epoch:{}'.format(epoch))
        losses = []
        pair_losses = []
        crf_losses = []
        attn_losses = []
        
        for j in trange(trainset.batch_count):
            
            _, _, _, review_embedder_input, reply_embedder_input, _, _, tags, review_biotags, reply_biotags, review_mask, reply_mask, reply_lengths, review_lengths = trainset.get_batch(j)
            
            """
            fill in -1 for attention template
            """
            attn_template = tags.clone().detach()
            for idx, (review_length, reply_length) in enumerate(zip(review_lengths, reply_lengths)):    
                for i in range(review_length):
                    attn_template[idx][i][:reply_length][attn_template[idx][i][:reply_length] == -1] = 0

            """
            dynamic negative sampling
            """
            for idx, (review_length, reply_length) in enumerate(zip(review_lengths, reply_lengths)):    
                for i in range(review_length):
                    if (tags[idx][i][:reply_length] == -1).sum().item() >= args.negative_sample:
                        neg_idx = (tags[idx][i][:reply_length] == -1).nonzero(as_tuple=False).view(-1)
                        choice = torch.multinomial(neg_idx.float(), args.negative_sample)
                        tags[idx][i][neg_idx[choice]] = 0
                    else:
                        tags[idx][i][:reply_length][tags[idx][i][:reply_length] == -1] = 0
            
            pair_logits, crf_loss, attn = model(review_embedder_input, reply_embedder_input, review_mask, reply_mask, review_lengths, reply_lengths, review_biotags, reply_biotags)
            logits_flatten = pair_logits.reshape(-1, pair_logits.size()[-1])
            tags_flatten = tags.reshape([-1])
            pair_loss = F.cross_entropy(logits_flatten, tags_flatten, ignore_index=-1, reduction='sum')
            attn_template += 1
            attn_template[attn_template==2] = -1
            attn_loss = torch.sum(attn*attn_template)
            if args.attention_loss:
                loss = args.pair_weight*pair_loss + crf_loss + args.attn_weight*attn_loss/sum([args.ema**i for i in range(args.iteration)])
            else:
                loss = args.pair_weight*pair_loss + crf_loss
            losses.append(loss.item())
            pair_losses.append(pair_loss.item()*args.pair_weight)
            crf_losses.append(crf_loss.item())
            attn_losses.append(attn_loss.item()*args.attn_weight/sum([args.ema**i for i in range(args.iteration)]))
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if args.optimizer == 'adamw':
                scheduler.step()
            model.zero_grad()

        print('average loss {:.4f}'.format(np.average(losses)))
        print('average pairing loss {:.4f}'.format(np.average(pair_losses)))
        print('average crf loss {:.4f}'.format(np.average(crf_losses)))
        print('average attention loss {:.4f}'.format(np.average(attn_losses)))
        print(colored('Evaluating dev set: ', color='red'))
        joint_precision, joint_recall, joint_f1, dev_attn_loss = eval(model, devset, args)
        print(colored('Evaluating test set: ', color='red'))
        _, _, _, test_attn_loss = eval(model, testset, args)

        attn_losses_all.append(np.average(attn_losses))
        attn_losses_dev.append(dev_attn_loss)
        attn_losses_test.append(test_attn_loss)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model_name + '.pt'
            torch.save(model, model_path)
            best_joint_precision = joint_precision
            best_joint_recall = joint_recall
            best_joint_f1 = joint_f1
            best_joint_epoch = epoch
    
    plot_attn_loss(args.epochs, args.model_name, attn_losses_all, attn_losses_dev, attn_losses_test, os.path.join(args.output_dir, args.model_name, f'{args.model_name}_losses.png'))

    print(colored('Final evluation on dev set: ', color='red'))
    print('best epoch: {}\tbest dev precision: {:.5f}\tbest dev recall: {:.5f}\tbest dev f1: {:.5f}\n\n'.format(best_joint_epoch, best_joint_precision, best_joint_recall, best_joint_f1))


def eval(model, dataset, args, output_results=False):
    
    model.eval()
    
    with torch.no_grad():
        
        all_ids = []
        all_preds = []
        all_labels = []
        all_review_lengths = []
        all_reply_lengths = []
        all_review_bio_preds = []
        all_reply_bio_preds = []
        all_review_bio_golds = []
        all_reply_bio_golds = []

        attn_losses = []
        
        for i in range(dataset.batch_count):
            sentence_ids, _, _, review_embedder_input, reply_embedder_input, _, _, tags, review_biotags, reply_biotags, review_mask, reply_mask, reply_lengths, review_lengths = dataset.get_batch(i)
            pair_logits, review_decode_idx, reply_decode_idx, attn = model.decode(review_embedder_input, reply_embedder_input, review_mask, reply_mask, review_lengths, reply_lengths)
            pair_preds = torch.argmax(pair_logits, dim=3)
            all_ids.extend(sentence_ids)
            all_preds.extend(pair_preds.cpu().tolist())
            all_labels.extend(tags.cpu().tolist())
            all_review_lengths.extend(review_lengths.cpu().tolist())
            all_reply_lengths.extend(reply_lengths.cpu().tolist())
            all_review_bio_golds.extend(review_biotags.cpu().tolist())
            all_reply_bio_golds.extend(reply_biotags.cpu().tolist())
            all_review_bio_preds.extend(review_decode_idx.cpu().tolist())
            all_reply_bio_preds.extend(reply_decode_idx.cpu().tolist())

            attn_template = tags.clone().detach()
            attn_template += 1
            attn_template[attn_template==2] = -1
            attn_loss = torch.sum(attn*attn_template)
            attn_losses.append(attn_loss.item()*args.attn_weight/sum([args.ema**i for i in range(args.iteration)]))


        metric = Metric(args, all_preds, all_labels, all_review_lengths, all_reply_lengths, all_review_bio_preds,
                        all_reply_bio_preds, all_review_bio_golds, all_reply_bio_golds)
        precision, recall, f1 = metric.score_uniontags()
        review_results = metric.score_review()
        reply_results = metric.score_reply()
        bio_results = metric.score_bio(review_results, reply_results)
        pair_results = metric.score_pair()
        print('Argument\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(bio_results[0], bio_results[1],
                                                                   bio_results[2]))
        print('Pairing\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(pair_results[0], pair_results[1],
                                                               pair_results[2]))
        print('Overall\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

        if output_results:
            writer = Writer(args, all_preds, all_labels, all_review_lengths, all_reply_lengths, all_review_bio_preds,
                        all_reply_bio_preds, all_review_bio_golds, all_reply_bio_golds)
            writer.output_results()

    model.train()
    return precision, recall, f1, np.average(attn_losses)


def visualize_attn(model, dataset, args):
    with torch.no_grad():
        if not args.plot_numbers:
            return
        elif len(args.plot_numbers) == 1 and args.plot_numbers[0] < 0:
            if -args.plot_numbers[0] > len(dataset):
                print(colored('no attention weights visualized because the number you specified is larger than the size of test dataset', color='red'))
                return
            else:
                ids = list(range(-args.plot_numbers[0]))
        else:
            ids = args.plot_numbers
        id = 0
        max_id = max(ids)
        for i in range(dataset.batch_count):
            if not (set(range(i * args.batch_size, min((i + 1) * args.batch_size, len(dataset)))) & set(ids)):
                id += min((i + 1) * args.batch_size, len(dataset)) - i * args.batch_size
                continue
            _, _, _, review_embedder_input, reply_embedder_input, _, _, tags, review_biotags, reply_biotags, review_mask, reply_mask, reply_lengths, review_lengths = dataset.get_batch(i)
            _, _, attns = model(review_embedder_input, reply_embedder_input, review_mask, reply_mask, review_lengths, reply_lengths, review_biotags, reply_biotags)
            """
            fill in -1 for attention template
            """
            attn_templates = tags.clone().detach()
            for idx, (review_length, reply_length) in enumerate(zip(review_lengths, reply_lengths)):    
                for i in range(review_length):
                    attn_templates[idx][i][:reply_length][attn_templates[idx][i][:reply_length] == -1] = 0
            for attn, attn_template, review, reply, review_length, reply_length in zip(attns, attn_templates, review_biotags, reply_biotags, review_lengths, reply_lengths):
                if id in ids:
                    plot_attention_weights(attn[:review_length, :reply_length].cpu(), [idx2labels[idx] for idx in review[:review_length].cpu().tolist()], [idx2labels[idx] for idx in reply[:reply_length].cpu().tolist()], os.path.join(args.output_dir, args.model_name, args.model_name + '_' + str(id) + '.' + args.img_format))
                    plot_attention_weights(attn_template[:review_length, :reply_length].cpu(), [idx2labels[idx] for idx in review[:review_length].cpu().tolist()], [idx2labels[idx] for idx in reply[:reply_length].cpu().tolist()], os.path.join(args.output_dir, args.model_name, args.model_name + '_' + str(id) + 'gold' + '.' + args.img_format))
                if id == max_id:
                    return
                id += 1
    return


def test(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.model_name)):
        os.makedirs(os.path.join(args.output_dir, args.model_name))

    print(colored('Final evluation on test set: ', color='red'))
    model_path = args.model_dir + args.model_name + '.pt'
    if args.device == 'cpu':
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args, is_train=False)
    testset = DataIterator(instances, args)
    eval(model, testset, args, output_results=True)
    visualize_attn(model, testset, args)
    
    """
    count number of parameters
    """
    num_param = 0
    for layer, weights in model.state_dict().items():
        if layer.startswith('embedder.bert'):
            continue
        prod = 1
        for dim in weights.size():
            prod *= dim
        num_param += prod
    print(colored('There are in total {} parameters within the model'.format(num_param), color='yellow'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="saved_models/",
                        help='model path prefix')
    parser.add_argument('--model_name', type=str, default="model1",
                        help='model name')
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='test dataset outputs directory')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="rr_submission_v3",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=210,
                        help='max length of a sentence')
    parser.add_argument('--max_bert_token', type=int, default=200,
                        help='max length of bert tokens for one sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument('--num_instances', type=int, default=-1,
                        help='number of instances to read')
    parser.add_argument('--test_code', type=bool, default=False,
                        help='to read train/dev/test or test/test/test')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='set random seed')
    parser.add_argument('--encoding_scheme', type=str, default='BIO', choices=['BIO', 'IOBES'],
                        help='encoding scheme for linear CRF')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-cased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-cased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--token_embedding', type=bool, default=True,
                        help='additional lstm embedding over pre-trained bert token embeddings')
    parser.add_argument('--freeze_bert', type=bool, default=True,
                        help='whether to freeze parameters of pre-trained bert model')
    parser.add_argument('--num_embedding_layer', type=int, default=1,
                        help='number of layers for token LSTM')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--embedding_dim', type=int, default=20,
                        help='dimension of type embedding')
    parser.add_argument('--layer_norm', type=bool, default=False,
                        help='whether apply layer normalization to RNN model')
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='whether use bidirectional 2D-GRU to RNN model')
    parser.add_argument('--attention', type=str, default='tanh', choices=['tanh', 'cosine_similarity'],
                        help='attention mechanism')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate') 
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm used during backprop') 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'],
                        help='optimizer choice') 
    parser.add_argument('--pair_weight', type=float, default=0.6,
                        help='pair loss weight coefficient for loss computation') 
    parser.add_argument('--attn_weight', type=float, default=2,
                        help='attention loss weight coefficient for loss computation')
    parser.add_argument('--ema', type=float, default=1.0,
                        help='EMA coefficient alpha')
    parser.add_argument('--lstm_share_param', type=bool, default=True,
                        help='whether to share same LSTM layer for review&reply embedding')
    parser.add_argument('--share_crf_param', type=bool, default=True,
                        help='whether to share same CRF layer for review&reply decoding')
    parser.add_argument('--cnn_classifier', type=bool, default=False,
                        help='whether to use cnn for pairing predictor')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--negative_sample', type=int, default=1000,
                        help='number of negative samples, 1000 means all') 
    parser.add_argument('--plot_numbers', nargs='+', type=int, default=[],
                        help='instance ids to visualize attention, emtpy means none, negative number -n means the first n instances')                    
    parser.add_argument('--img_format', type=str, default='png', choices=['png', 'pdf'],
                        help='image format') 

    parser.add_argument('--pair_threshold', type=float, default=0.5,
                        help='pairing threshold during evaluation')
    parser.add_argument('--iteration', type=int, default=2,
                        help='cross module iteration')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel size for CNN module')
    parser.add_argument('--attention_loss', type=bool, default=True,
                        help='whether to include attention loss')
    parser.add_argument('--cross_update', type=bool, default=True,
                        help='whether to cross update the sequence embedding')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='training epoch number')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
