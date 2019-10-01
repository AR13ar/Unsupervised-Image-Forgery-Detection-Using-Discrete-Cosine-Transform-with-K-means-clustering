# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 04:51:21 2019

@author: Aditya Raj
"""
import numpy as np
from scipy.fftpack import dct
from sklearn.cluster import KMeans


def dct_block(image):
#dividing the gray-scale image into overlapping blocks of size 8x8
    filt_dim = 10    #filt_dim signifies the size of the window of the filter which will move on the image
    image = np.pad(image,int(filt_dim/2), mode = 'wrap' )
    a,b = image.shape       #image.shape->reading the size of the image
    win_data = []
    stride = 2      #stride means how many steps the filter moves at a time i.e defines the overlapping
    for i, row in enumerate(range (0,a-filt_dim,stride)):
        for j, col in enumerate(range(0, b-filt_dim,stride)):
            win = image[row:row+filt_dim,col:col+filt_dim]
            win_data.append(win.flatten())   #append means adding to the list
    win_data = np.array(win_data)            #np.array converts the list into 2-D array form
    #applying DCT on the image
    dct_list=[]
    for i in range(win_data.shape[0]):    
        dct_list.append(dct(np.reshape(win_data[i],(filt_dim,filt_dim)), 1))
    dct_list=np.array(dct_list)
    return dct_list, win_data


def kmeans(dct_coeff_list_16):
#applying KMeans
    labels = []       #to store labels of the clusters
    center = []       #to store centre of each cluster
    k_dct = np.array(dct_coeff_list_16)           #converting 16 coeff list to K_dct array
    k_dct = k_dct.reshape(k_dct.shape[0], 16)     #appending the 16 dct coeffs from all the blocks into K_dct array 
    k_dct = (k_dct-k_dct.mean())/k_dct.std()
    clf=KMeans(20)                               #no. of clusters
    clf.fit(k_dct)    
    labels.append( clf.labels_)            #finding the labels
    center.append(clf.cluster_centers_)    #finding the centres 
    center = np.array(center)
    center = center.reshape(center.shape[1], center.shape[2])   
    labels = np.array(labels)
    labels = labels.T                  #taking transpose of the labels matrix
    return center, labels, k_dct 

