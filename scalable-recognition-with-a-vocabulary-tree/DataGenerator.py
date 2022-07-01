from os.path import join
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import cbir

import matplotlib.pyplot as plt
import os

#超参数
n_branches = 4
depth = 4
def load_database():
	# initialise the database
	# 加载所有图像路径
	dataset = cbir.Dataset(folder="cbir/data/jpg")
	orb = cbir.descriptors.Orb()
	# Let's create the vocabulary tree
	voc = cbir.encoders.VocabularyTree(n_branches=n_branches, depth=depth, descriptor=orb)
	# Now, let's extract the features
	# This takes about 10 minutes on a personal computer.
	# We have done it already for you and saved all the features in `data/features_orb.hdf5`. If you want to do it
	# from scratch, just delete or rename that file.
	features = voc.extract_features(dataset)
	# We can now construct the tree using the extracted features
	voc.fit(features)

	# now that we have a tranied encoder, let's create our database for retrieval
	db = cbir.Database(dataset, encoder=voc)

	# save the database on disk for later
	isLoad = db.load()
	print("Load ok:", isLoad)

	# query = "104000.jpg"
	# scores = db.retrieve(query)
	# db.show_results(query, scores, figsize=(30, 10))
	return db,dataset

# next stage: train data generate: each img feature query once, top-K as positive, random M in database as negatives
def write_to_file(num,df_data,basePath='dssm_retrieval/data/train'):
	path = os.path.join(basePath,'img_features'+num+'.csv')

	# 解决追加模式写的表头重复问题
	if not os.path.exists(path):
		df_data.to_csv(path, sep='\t',header=["query","target","label"], index=False, mode='a')
	else:
		df_data.to_csv(path, sep='\t',header=False, index=False, mode='a')

# retrieve each img positives and negatives
def generate_train_test(db,dataset,other_features,posNum=2,negNum=10):
	train_data=pd.DataFrame(columns=("query","target","label"))
	db.shuffule_image_path()
	num=0
	for i, imgName in enumerate(dataset.image_paths):
		img_embeding=db.database[imgName]
		scores = db.retrieve(imgName)
		for j in range(min(posNum,len(scores))):
			target_embeding=db.embedding(scores[j])
			together_embeding=target_embeding.extend(other_features[scores[j]])
			train_data=train_data.append(({'query':img_embeding,'target':together_embeding,'label':1}), ignore_index=True)
		# random negetive
		for k in range(negNum):
			target_embeding=db.embedding(db.dataset.image_paths[k])
			together_embeding = target_embeding.extend(other_features[db.dataset.image_paths[k]])
			train_data= train_data.append(({'query':img_embeding,'target':together_embeding,'label':0}), ignore_index=True)
		if (i+1) % 100000==0:
			trains, tests = train_test_split(train_data, test_size=0.3)
			# write disk
			write_to_file(num,trains,basePath='dssm_retrieval/data/train')
			write_to_file(num, tests, basePath='dssm_retrieval/data/test')
			num+=1
			train_data = pd.DataFrame(columns=("query", "target", "label"))
	if len(train_data)>0:
		trains, tests = train_test_split(train_data, test_size=0.3)
		# write disk
		write_to_file(num, trains, basePath='dssm_retrieval/data/train')
		write_to_file(num, tests, basePath='dssm_retrieval/data/test')
# 加载数据库特征库
db,dataset=load_database()
print("start extracting other features...")
# extract color,gabor,xxx features
other_features=db.dataset.extract_other_features()
print("end extracting other features!")
generate_train_test(db,dataset,other_features)