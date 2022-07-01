import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

from dssm_retrieval.reader.ImgDataLoader import config

CHAR_SIZE=21128
EPOCH=10
BATCH_SIZE=50
LR=0.00015
class DSSM(torch.nn.Module):
	def __init__(self):
		super(DSSM, self).__init__()
		self.embedding = nn.Embedding(CHAR_SIZE, config.embedding_size)
		self.linear1 = nn.Linear(config.embedding_size, 256)
		self.linear2 = nn.Linear(256, 128)
		self.linear3 = nn.Linear(128, 64)
		self.dropout = nn.Dropout(p=0.2)

	def forward(self, a, b):
		# 将各索引id的embedding向量相加
		a = self.embedding(a).sum(1)
		b = self.embedding(b).sum(1)

		a = torch.tanh(self.linear1(a))
		a = self.dropout(a)
		a = torch.tanh(self.linear2(a))
		a = self.dropout(a)
		a = torch.tanh(self.linear3(a))
		a = self.dropout(a)

		b = torch.tanh(self.linear1(b))
		b = self.dropout(b)
		b = torch.tanh(self.linear2(b))
		b = self.dropout(b)
		b = torch.tanh(self.linear3(b))
		b = self.dropout(b)

		cosine = torch.cosine_similarity(a, b, dim=1, eps=1e-8)
		return cosine

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)