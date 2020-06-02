# -*- coding:utf-8
import numpy as np
import os
import cv2

ROOT_DIR = '/root/workspace/system-check/'
DATA_PATH = ROOT_DIR + 'data/olivetti/'

def size_to_img(x):
    x = x.reshape((-1, 32, 32, 3))
    x = x.transpose(0, 3, 1, 2)
    return x

def load_data():
    imgs = np.load(DATA_PATH+'olivetti_faces.npy')
    new_imgs = np.zeros((len(imgs), 32, 32))
    for i in range(len(imgs)):
        new_imgs[i,:,:] = cv2.resize(imgs[i], (32, 32))
    new_imgs = np.stack((new_imgs,)*3, axis=-1)
    new_imgs = np.reshape(new_imgs, (len(new_imgs), 3*32*32))
    labels = np.load(DATA_PATH+'olivetti_faces_target.npy')
    return new_imgs.astype('float32'),labels.astype('int32')

def load_train_data():
    data, labels = load_data()
    x, y = [], []
    for i in range(len(data)):
        if i%10 < 7:
            x.append(data[i,:])
            y.append(labels[i])
    # 利用np.concatenate()函数将多个list拼接成一个list
    return np.asarray(x), np.asarray(y)

def load_test_data():
    data, labels = load_data()
    x, y = [], []
    for i in range(len(data)):
        if i%10 >= 7:
            x.append(data[i,:])
            y.append(labels[i])
    return np.asarray(x), np.asarray(y)

def load_olivetti():
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()
    X_train = size_to_img(X_train)
    X_test = size_to_img(X_test)
    return X_train, y_train, X_test, y_test