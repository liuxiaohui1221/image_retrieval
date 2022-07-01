import cbir
import numpy as np
import matplotlib.pyplot as plt

#超参数
n_branches = 4
depth = 4
# initialise the database
def init_db():
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

	# And we can index all the images
	# you can use db.index() to index all the images at once
	db.index()

	# save the database on disk for later
	db.save()

	query = "104000.jpg"
	scores = db.retrieve(query)
	db.show_results(query, scores, figsize=(30, 10))

init_db()
