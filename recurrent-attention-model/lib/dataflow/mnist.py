#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

#import os
#import gzip
#import struct

#from datetime import datetime
#import numpy as np

# from tensorcv.dataflow.base import RNGDataFlow

#_RNG_SEED = None

#def get_rng(obj=None):
#    """
#    This function is copied from `tensorpack
#    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
#    Get a good RNG seeded with time, pid and the object.
#
#    Args:
#        obj: some object to use to generate random seed.
#
#    Returns:
#        np.random.RandomState: the RNG.
#    """
#    seed = (id(obj) + os.getpid() +
#            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
#    if _RNG_SEED is not None:
#        seed = _RNG_SEED
#    return np.random.RandomState(seed)


#def identity(im):
#    return im

#class MNISTData(object):
#    """ class for MNIST dataflow

#        To access the data of mini-batch, first get data of all the channels
#        through batch_data = MNISTData.next_batch_dict()
#        then use corresponding key to get label or image through
#        batch_data[key].
#    """
#    def __init__(self, name, data_dir='', n_use_label=None, n_use_sample=None,
#                 batch_dict_name=None, shuffle=True, pf=identity):
#        """
#        Args:
#            name (str): name of data to be read (['train', 'test', 'val'])
#            data_dir (str): directory of MNIST data
#            n_use_label (int): number of labels to be used
#            n_use_sample (int): number of samples to be used
#            batch_dict_name (list of str): list of keys for 
#                image and label of batch data
#            shuffle (bool): whether shuffle data or not
#            pf: pre-process function for image data
#        """
#        assert os.path.isdir(data_dir)
#       self._data_dir = data_dir

#        self._shuffle = shuffle
#        self._pf = pf
#        if not isinstance(batch_dict_name, list):
#            batch_dict_name = [batch_dict_name]
#        self._batch_dict_name = batch_dict_name

#        assert name in ['train', 'test', 'val']
#        self.setup(epoch_val=0, batch_size=1)
#
#        self._load_files(name, n_use_label, n_use_sample)
#        self._image_id = 0
#
#    def next_batch_dict(self):
#        batch_data = self.next_batch()
#        data_dict = {key: data for key, data
#                     in zip(self._batch_dict_name, batch_data)}
#        return data_dict
#    
#    def read_img(self,path):
#        img = cv2.imread(path,0)
#        img = img.astype(dtype=np.float32)/256.0
#        return img
#
#    def _load_files(self, name, n_use_label, n_use_sample):
#        if name=='train':
#            directory = 'C:/Users/Rishikesh/Documents/Codes/IITM_neural_network_Assignment/recurrent-attention-model/example/yale b/ExtendedYaleB/train'#directory for training data
#        if name=='val':
#            directory = 'C:/Users/Rishikesh/Documents/Codes/IITM_neural_network_Assignment/recurrent-attention-model/example/yale b/ExtendedYaleB/test'
#        data = {0:{}}
#        labels = []
#        seed = 22345 #seed for the random state
#        #read the images from the directory
#        for subdir, dirs, files in os.walk(directory):
#            for image in files:
#               r = self.read_img(directory+'/'+image)
#                s = image.split('_')
#                #adding this line for yale            
#                s = s[0].split('B')
#                if int(s[1]) not in labels:
#                    labels.append(int(s[1]))
#                try:
#                    data[0][int(s[1])].append([r,int(s[1])])
#                except KeyError:
#                    data[0][int(s[1])] = [[r,int(s[1])]]
#        np.random.RandomState(seed)
#        train_set = list(); 
#        #give the real label to the list
#        for k,v in data.items():
#            for k2, v2 in v.items():
#                temp = []
#                for item in data[k][k2]:
#                    temp.append((item[0],labels.index(item[1])))
#                np.random.shuffle(temp)    
#                data[k][k2] = temp
#                for image_tuple in data[k][k2][:]:
#                    train_set.append(image_tuple)    
#        np.random.RandomState(seed)
#        np.random.shuffle(train_set)
#        print (len(train_set))
#        image_list = np.fromstring(train_set,dtype=np.uint8)
#        label_list = np.formstring(labels, dtype=np.uint8)
#        self.label_list = np.array(label_list)
#        self.im_list = np.array(image_list)


#    def _suffle_files(self):
#        if self._shuffle:
#            idxs = np.arange(self.size())

#            self.rng.shuffle(idxs)
#            self.im_list = self.im_list[idxs]
#            self.label_list = self.label_list[idxs]
#
#    def size(self):
#        return self.im_list.shape[0]

#    def next_batch(self):
#        assert self._batch_size <= self.size(), \
#          "batch_size {} cannot be larger than data size {}".\
#           format(self._batch_size, self.size())
#        start = self._image_id
#        self._image_id += self._batch_size
#        end = self._image_id
#        batch_files = self.im_list[start:end]
#        batch_label = self.label_list[start:end]
#
#        if self._image_id + self._batch_size > self.size():
#            self._epochs_completed += 1
#            self._image_id = 0
#            self._suffle_files()
#        return [batch_files, batch_label]
#
#    def setup(self, epoch_val, batch_size, **kwargs):
#        self._epochs_completed = epoch_val
#        self._batch_size = batch_size
#        self.rng = get_rng(self)
#        try:
#            self._suffle_files()
#        except AttributeError:
#            pass
#

#!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # File: mnist.py
# # Author: Qian Ge <geqian1001@gmail.com>

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np 
import pandas
from PIL import Image
from .utilpack import get_rng
from .DataFlow import DataFlow

class RNGDataFlow(DataFlow):
    def _reset_state(self):
        self.rng = get_rng(self)

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self.file_list = self.file_list[idxs]

    def suffle_data(self):
        self._suffle_file_list()

def identity(im):
    return im

def concat_pair(im_1, im_2):
    im_1 = np.expand_dims(im_1, axis=-1)
    im_2 = np.expand_dims(im_2, axis=-1)
    return np.concatenate((im_1, im_2), axis=-1)

def get_mnist_im_label(name, mnist_data):
    if name == 'train':
        return mnist_data.train.images, mnist_data.train.labels
    elif name == 'val':
        return mnist_data.validation.images, mnist_data.validation.labels
    else:
        return mnist_data.test.images, mnist_data.test.labels

class MNISTData(RNGDataFlow):
    def __init__(self, name, batch_dict_name=None, data_dir='', shuffle=True, pf=identity):
        assert os.path.isdir(data_dir)
        self._data_dir = data_dir

        self._shuffle = shuffle
        if pf is None:
            pf = identity
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        assert name in ['train', 'test', 'val']
        self.setup(epoch_val=0, batch_size=1)

        self._load_files(name)
        self._image_id = 0
    
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    
    def setup(self, epoch_val, batch_size, **kwargs):
        self._epochs_completed = epoch_val
        self._batch_size = batch_size
        self.rng = get_rng(self)
        try:
            self._suffle_files()
        except AttributeError:
            pass

    def read_img(self,path):
        img = cv2.imread(path,0);
        img = img.astype(np.float32)
        img = img/255.
        return img


    def next_batch_dict(self):
        batch_data = self.next_batch()
        data_dict = {key: data for key, data in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _load_files( self,name):
        if name=='train':
            Dirs = 'C:/Users/Rishikesh/Documents/Codes/yale b/ExtendedYaleB/'#directory for training data
        if name=='val':
            Dirs = 'C:/Users/Rishikesh/Documents/Codes/yale b/ExtendedYaleB/train'
        Nclass = 0
        (Images, Labels, Names, Paths, ID) = ([], [], [], [], 0)
        for (_,Dirs,_) in os.walk(Dirs):
            for SubDirs in Dirs:
                SubjectPath = os.path.join(Dirs,SubDirs)
                Nclass+=1          
                for FileName in os.listdir(SubjectPath):
                    path = SubjectPath + "/" + FileName
                    Img = np.array(self.read_img(path))#misc.imread(path, mode='L')
                    Paths.append(path)
                    (img_row, img_col) = Img.shape
                    Labels.append(int(FileName))
                    Images.append(Img)
                    Img = np.resize(28,28)
        Images = np.asarray(Images, dtype='float32').reshape([-1,img_row, img_col, 1])
        lbls =[]
        for label in Labels:
            lbls.append(Categorical_([label],Nclass)[0])
        self.im_list = Images
        self.label_list = lbls
#        data = {0:{}}
#        labels = []
#        seed = 22345 #seed for the random state
#        #read the images from the directory
#        for subdir, dirs, files in os.walk(directory):
#            for image in files:
#                r = self.read_img(directory+'/'+image)
#                s = image.split('_')
#                #adding this line for yale            
#                s = s[0].split('B')
#                if int(s[1]) not in labels:
#                    labels.append(int(s[1]))
#                try:
#                    data[0][int(s[1])].append([r,int(s[1])])
#                except KeyError:
#                    data[0][int(s[1])] = [[r,int(s[1])]]
#        np.random.RandomState(seed)
#        train_set = list(); 
#        #give the real label to the list
#        for k,v in data.items():
#            for k2, v2 in v.items():
#                temp = []
#                for item in data[k][k2]:
#                    temp.append((item[0],labels.index(item[1])))
#                np.random.shuffle(temp)    
#                data[k][k2] = temp
#                for image_tuple in data[k][k2][:]:
#                    train_set.append(image_tuple)    
#        np.random.RandomState(seed)
#        np.random.shuffle(train_set)
#        print (len(train_set))
#        self.label_list = np.array(labels)
#        self.im_list = np.array(train_set)

#    def _load_files(self, name):
#        mnist_data = input_data.read_data_sets(self._data_dir, one_hot=False)
#        self.im_list = []
#        self.label_list = []

#         mnist_images, mnist_labels = get_mnist_im_label(name, mnist_data)
#         for image, label in zip(mnist_images, mnist_labels):
#             # TODO to be modified
#             image = np.reshape(image, [28, 28, 1])
            
#             # image = np.reshape(image, [28, 28, 1])
            
#             self.im_list.append(image)
#             self.label_list.append(label)
#         self.im_list = np.array(self.im_list)
#         self.label_list = np.array(self.label_list)

#         self._suffle_files()

    def _suffle_files(self):
        if self._shuffle:
            idxs = np.arange(self.im_list.shape[0])

            self.rng.shuffle(idxs)
            self.im_list = self.im_list[idxs]
            self.label_list = self.label_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size {} cannot be larger than data size {}".\
        format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = []
        for im in self.im_list[start:end]:
            im = np.reshape(im, [28, 28])
            im = self._pf(im)
            im = np.expand_dims(im, axis=-1)
            batch_files.append(im)

        batch_label = self.label_list[start:end]

        if self._image_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._image_id = 0
            self._suffle_files()
        return [batch_files, batch_label]


# class MNISTPair(MNISTData):
#     def __init__(self,
#                  name,
#                  label_dict,
#                  batch_dict_name=None,
#                  data_dir='',
#                  shuffle=True,
#                  pf=identity,
#                  pairprocess=concat_pair,
#                  ):
#         self._pair_fnc = pairprocess
#         self._label_dict = label_dict
#         super(MNISTPair, self).__init__(name=name,
#                                         batch_dict_name=batch_dict_name,
#                                         data_dir=data_dir,
#                                         shuffle=shuffle,
#                                         pf=pf,)

#     def size(self):
#         return int(np.floor(self.im_list.shape[0] / 2.0))

#     def next_batch(self):
#         assert self._batch_size <= self.size(), \
#           "batch_size {} cannot be larger than data size {}".\
#            format(self._batch_size, self.size())
#         # start = self._image_id
#         # self._image_id += self._batch_size * 2
#         # end = self._image_id
#         batch_files = []
#         batch_label = []
#         start = self._image_id
#         for data_id in range(0, self._batch_size):
#             im_1 = np.reshape(self.im_list[start], [28, 28])
#             im_2 = np.reshape(self.im_list[start + 1], [28, 28])
#             im = self._pair_fnc(im_1, im_2)
#             im = np.expand_dims(im, axis=-1)
#             batch_files.append(im)

#             label_1 = self.label_list[start]
#             label_2 = self.label_list[start + 1]
#             label = self._label_dict['{}{}'.format(label_1, label_2)]
#             batch_label.append(label)
#             start = start + 2
#         end = start
#         self._image_id = end

#         if self._image_id + self._batch_size > self.size():
#             self._epochs_completed += 1
#             self._image_id = 0
#             self._suffle_files()
#         return [batch_files, batch_label]
