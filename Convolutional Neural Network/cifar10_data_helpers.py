
# coding: utf-8

# In[1]:

import numpy as np
import pickle
import os
import download_dataset
from one_hot_encoded import one_hot_encoded


# In[2]:

# directory for downloading the CIFAR-10 dataset
data_path = "data/CIFAR-10/"

# url for downloading the dataset
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# In[2]:

# Data dimensions

# CIFAR-10 images are 32 pixels in each dimension
img_size = 32

# no of color channels in each images: 3 - for RGB
num_channels = 3

# Length of an image when flattened to 1-dim array.
img_size_flat = img_size * img_size * num_channels

# no of classes; cifar-10 has 10 classes
num_classes = 10


# In[1]:

# CIFAR-10 has 50000 training images and 10000 test images

# divide the training samples into 5 batches
num_train_batches = 5

# no of images for each batch in the training set
images_per_train_batch = 10000

# total no of images in the training set
num_images_train = num_train_batches * images_per_train_batch


# In[5]:

# private functions for downloading, unpacking, and loading data files

# full path of a data file for the dataset
def get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

# unpickle the data
def unpickle(filename):
    file_path = get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode="rb") as file:
        # in python3, set the encoding to remove exception error
        data = pickle.load(file, encoding="bytes")
        
    return data


# In[6]:

# Convert images from the CIFAR-10 format and return a 4-dim array of shape: [images_number, height, width, channel]
# where the pixels are floats between 0.0 and 1.0

def convert_images(raw):

    # convert the raw images from the data-files to floating points
    raw_float = np.array(raw, dtype=float) / 255.0

    # reshape the array to 4-dims
#     images = np.reshape(raw_float, (-1, num_channels, img_size, img_size))
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    
    # reorder the indices of the array using transpose
#     images = np.transpose(images, (0, 2, 3, 1))
    images = images.transpose([0, 2, 3, 1])
    
    return images


# In[7]:

# Load the pickled data file from the CIFAR-10 dataset
# Return the converted images and the class number for each image

def load_data(filename):

    # load the pickled data file
    data = unpickle(filename)

    # get the raw images
    raw_images = data[b'data']

    # get the class-numbers for each image. convert to numpy array
    cls = np.array(data[b'labels'])

    # convert the images
    images = convert_images(raw_images)

    return images, cls


# In[8]:

# Download and extract the data

def download_and_extract():
    download_dataset.download_and_extract(url=data_url, download_dir=data_path)


# In[9]:

# Load the names for the classes in the CIFAR-10 dataset

def load_class_names():
    
    # load the class names from the pickled file
    raw = unpickle(filename="batches.meta")[b'label_names']
    
    # convert from binary strings
    names = [x.decode("utf-8") for x in raw]
    
    return names


# In[10]:

# Load all the training data for the CIFAR-10 dataset
# The dataset is split into 5 data files which are merged here
# return the images, class numbers and one-hot encoded class labels

def load_training_data():

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    print(images.shape)
    cls = np.zeros(shape=[num_images_train], dtype=int)
    print(cls.shape)

    # begin index for the current batch
    begin = 0
    
    # for each data file in the training batch
    for i in range(num_train_batches):
        
        # Load the images and class numbers from the data-file.
        images_batch, cls_batch = load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end
        
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


# In[11]:

# Load all the test data for the CIFAR-10 dataset

def load_test_data():
    images, cls = load_data(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

