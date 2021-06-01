from collections import defaultdict
import itertools
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dev",
                        help='dataset train/dev/test')
args = parser.parse_args()
dataset = args.dataset

filein = open(f'{dataset}.txt','r').readlines()
file_list = []
inst = {}
inst_idx = 0
end_idx = -1
# review = defaultdict(list)
# reply = defaultdict(list)
passage = ''
passage_idx = 0
triples_list = []
sent_list = []
labelrr = []
label = []
pair = set()
reply_argu = set()
review_argu = set()
split_idx = -1
cnt = 0
for idx, line in enumerate(filein):
	line = line.strip()
	if not line:
		# assert len(review) == len(reply) 'length not equal'
		# print(review, reply)
		inst['id'] = str(passage_idx)
		passage_idx += 1
		inst['sentence'] = passage[:-11]
		# inst['triples'].append(triples_list)
		# pair_num = len(pair)

		triples_list = []
		# print(pair)
		pair = reply_argu.union(review_argu)
		# print(reply_argu,review_argu)
		# break
		if len(reply_argu) != len(pair):
			cnt+=1
		# count = 0
		for i, pair_idx in enumerate(list(pair)):
			review_seq = ''
			reply_seq = ''
			triples = {}
			# print(len(sent_list))
			# print(labelrr)
			count = 0
			for token_idx, token in enumerate(sent_list):
				if label[token_idx][2:] == pair_idx[2:]:
					# print(labelrr[token_idx][2:])
					if labelrr[token_idx][2:]=='Review':
						review_seq += token
						review_seq += '\\'
						review_seq += labelrr[token_idx][0]
						review_seq += '<tag>'
						reply_seq += token
						reply_seq += '\\O<tag>'
					elif labelrr[token_idx][2:]=='Reply':
						reply_seq += token
						reply_seq += '\\'
						reply_seq += labelrr[token_idx][0]
						reply_seq += '<tag>'
						review_seq += token
						review_seq += '\\O<tag>'
					else:
						print(labelrr[token_idx][2:])

				else:
					review_seq += token
					review_seq += '\\O<tag>'
					reply_seq += token
					reply_seq += '\\O<tag>'

			triples['uid']=pair_idx[2:]
			triples['target_tags'] = review_seq[:-5]
			triples['opinion_tags'] = reply_seq[:-5]
			triples['sentiment'] = 'neutral'
			triples_list.append(triples)
			
		inst['triples'] = triples_list
		inst['split_idx'] = split_idx
		file_list.append(inst)
		inst_idx = 0
		inst = {}
		# review = defaultdict(list)
		# reply = defaultdict(list)
		end_idx = -1
		split_idx = -1
		sent_list = []
		labelrr = []
		label = []
		pair = set()
		reply_argu = set()
		review_argu = set()
		passage = ''
		continue
		# break

	line = line.split('\t')
	sent = line[0]
	passage += sent
	passage += ' <sentsep> '

	sent_list.append(sent)
	labelrr.append(line[1])
	label.append(line[2])

	if line[1] == 'B-Reply':
		reply_argu.add(line[2])
	if line[1] == 'B-Review':
		review_argu.add(line[2])
	if line[-2] == 'Review':
		split_idx += 1


with open(f'{dataset}.json', 'w') as f_out:
	json.dump(file_list, f_out)

