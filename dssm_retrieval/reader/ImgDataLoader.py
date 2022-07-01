import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader,Dataset
import os
class config:
	max_seq_len = 768
	embedding_size = 768

class ImgReaderDataset(Dataset):
	def __init__(self, filepath):
		self.path = filepath
		self.max_seq_len=config.max_seq_len
		self.a_index, self.b_index, self.label = self.load_data(self.path)

	def __len__(self):
		return len(self.a_index)

	def __getitem__(self, idx):
		return self.a_index[idx], self.b_index[idx], self.label[idx]

	def load_data(self,filename):
		df=pd.read_csv(filename, sep='\t', names=["query", "target", "label"])
		q_a = df['query'].values
		t_b = df['target'].values
		label = df['label'].values
		return np.array(q_a), np.array(t_b), np.array(label)


