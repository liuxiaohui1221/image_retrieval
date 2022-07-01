import cv2
import numpy as np
from matplotlib import pyplot as plt

import cbir

# Loading the dataset
dataset = cbir.Dataset(folder="cbir/data/jpg")
descriptor = cbir.descriptors.Orb()
# Plotting the image
# plt.figure(figsize=(10,10))
# dataset.show_image(img)
# plt.title("Random image from the dataset")
# plt.show()

# Loading and displaying an image
img = dataset.read_image('110901.jpg')
# img=dataset.get_random_image()
# Plotting functions
# plt.figure(figsize=(12,4))
# plt.subplot(121)
# dataset.show_image(img, gray=True)
# plt.title('110901.jpg')
# plt.subplot(122)
# dataset.show_image(z, gray=True)
# plt.title("$\sin(I/10)$")
# plt.show()

cbir.marking.this_is_an_action_cell()


# Corner Detector parameters

# --- Add your code here ---
block_size = 11
kernel_size = 11
aperture_parameter = 0.01
treshold = 0.001
# --- End of custom code ---
# --- Plotting - do not remove the code below ---
# gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# dst = cv2.cornerHarris(gray,
#                        blockSize=block_size,
#                        ksize=kernel_size,
#                        k=aperture_parameter)
# dst = dst > treshold*dst.max()
# plt.figure(figsize=(15,10))
# descriptor.show_corners_on_image(gray,dst)
# plt.show()

def find_keypoints(image):
    ''' This function should take an image as input and return a list
    of keypoints as output
    '''
    # Creating the detector and setting some properties
    orb = cv2.ORB.create()
    # Detecting the keypoints
    keypoints = orb.detect(image)
    return keypoints

keypoints = find_keypoints(img)
# plt.figure(figsize=(15,10))
# Only displaying every 10 keypoints
img2 = cv2.drawKeypoints(img, keypoints[::10], None, color=(0,255,0), flags=4)
dataset.show_image(img2)
plt.show()

def extract_descriptors(image, keypoints):
    ''' This function should take an image as input and return a list
    of keypoints as output
    '''
    orb = cv2.ORB.create(1500, nlevels=32)
    keypoints, features = orb.compute(image, keypoints)
    return features
keypoints = find_keypoints(img)
features = extract_descriptors(img, keypoints)
print("Descriptor for the 1st keypoint:\n {}".format(features[0]))
print("Descriptor for the 2nd keypoint:\n {}".format(features[1]))
print("Descriptor for the 3rd keypoint:\n {}".format(features[2]))
print("Descriptor for the 4th keypoint:\n {}".format(features[3]))
print("Descriptor for the 5th keypoint:\n {}".format(features[4]))

keypoints = find_keypoints(img)
patches = descriptor.extract_patches(img, keypoints)
features = extract_descriptors(img, keypoints)
descriptor.show_random_descriptors(img, keypoints, patches, features)
plt.show()


# 3.1 Building the tree using hierarchical k-means
# !dot -c
import cbir
import numpy as np

n_branches = 2
depth = 3
orb = cbir.descriptors.Orb()
# initialise the database
voc = cbir.encoders.VocabularyTree(n_branches=n_branches, depth=depth, descriptor=orb)
# perform hierarchical k-means clustering on the features
voc.fit(np.random.randn(150, 1))
# plot the graph
# fig = voc.draw(figsize=(20, 7), labels=voc.nodes)
# plt.show()


# 3.2 Indexing the database using TF-IDF
import matplotlib.pyplot as plt

# let's take a look at how we can encode an image
image_id = "110901"
# as a graph
# voc.subgraph(image_id)
# print("Image embedding as graph:")
# plt.show()

# and its corresponding vector
embedding = voc.embedding(dataset.read_image(image_id))
print("\nImage embedding as vector:", embedding, "\n")
# fig = plt.figure(figsize=(15, 3))
# plt.bar(np.arange(len(embedding)), embedding)
# plt.gca().set_title("TF-IDF")
# plt.show()


# Chapter 4. Scoring and Retrieving (the online phase)
# take two images by using their ids
image_id_1 = "104000"
image_id_2 = "101000"

# read the images
image1 = dataset.read_image(image_id_1)
image2 = dataset.read_image(image_id_2)

# and then get their embeddings
em_1 = voc.embedding(image1)
em_2 = voc.embedding(image2)

print("\nImage 1 embedding:", em_1)
print("\n\nImage 2 embedding:", em_2)

fig, ax = plt.subplots(1, 2, figsize=(30, 3))
ax[0].bar(np.arange(len(em_1)), em_1)
ax[0].set_title("Image 1 TF-IDF")
ax[1].bar(np.arange(len(em_2)), em_2)
_ = ax[1].set_title("Image 2 TF-IDF")
plt.show()
def score(x1, x2):
    """Returns the euclidean distance between two tensors"""
    # --- Add your code here ---
    rmse = np.sqrt(np.mean(np.square(x1 - x2)))
    # --- End of custom code ---
    return rmse
print("score(rmse):",score(em_1,em_2))