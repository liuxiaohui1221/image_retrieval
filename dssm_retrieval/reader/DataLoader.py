import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader,Dataset
import os
class config:
	vocabPath = os.path.split(os.path.realpath(__file__))[0] + '/vocab.txt'
	tokenizer = BertTokenizer.from_pretrained(vocabPath)
	max_seq_len = 768
	embedding_size = 768

def load_vocab(vocabPath):
	vocab = open(vocabPath, encoding='utf-8').readlines()
	slice2idx = {}
	idx2slice = {}
	cnt = 0
	for char in vocab:
		char = char.strip('\n')
		slice2idx[char] = cnt
		idx2slice[cnt] = char
		cnt += 1
	return slice2idx, idx2slice

class DuReaderDataset(Dataset):
	def __init__(self, filepath):
		self.path = filepath
		self.tokenizer=config.tokenizer
		self.max_seq_len=config.max_seq_len
		self.a_index, self.b_index, self.label = self.load_char_data(self.path)

	def __len__(self):
		return len(self.a_index)

	def __getitem__(self, idx):
		return self.a_index[idx], self.b_index[idx], self.label[idx]

	def char_index(self,text_a, text_b):
		# slice2idx, idx2slice = load_vocab(vocabPath)
		a_list, b_list = [], []

		# 对文件中的每一行
		for a_sentence, b_sentence in zip(text_a, text_b):
			a = self.tokenizer.encode(str(a_sentence), max_length=self.max_seq_len, pad_to_max_length=True,
								 add_special_tokens=True)
			b = self.tokenizer.encode(str(b_sentence), max_length=self.max_seq_len, pad_to_max_length=True,
								 add_special_tokens=True)
			a_list.append(a)
			b_list.append(b)
		return a_list, b_list
	def load_char_data(self,filename):
		df = pd.read_csv(filename, sep='\t', names=["query", "null", "doc", "label"])
		text_a = df['query'].values
		text_b = df['doc'].values
		label = df['label'].values

		a_index, b_index = self.char_index(text_a, text_b)
		return np.array(a_index), np.array(b_index), np.array(label)


