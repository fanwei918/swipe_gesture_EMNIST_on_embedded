#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:37:54 2018

@author: weif
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras import backend as K
import numpy as np
from keras.models import load_model
# from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

batch_size = 128
num_classes = 27
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28








#%%
# EMNIST letters: 
from scipy.io import loadmat
emnist = loadmat('emnist-letters.mat')
# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)
# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]
# load test dataset# load  
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)
# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]
# store labels for visualization
#train_labels = y_train
#test_labels = y_test
# reshape using matlab order
x_train = x_train.reshape(x_train.shape[0], 28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0], 28, 28,order="A")


#    # getting c, s, w, which are : 1 swipt; cap/non-cap are similar; not similar with digits
x_train_letter_c =  x_train[y_train[:,0]==3] 
y_train_letter_c =  y_train[y_train[:,0]==3] 
x_test_letter_c =  x_test[y_test[:,0]==3] 
y_test_letter_c =  y_test[y_test[:,0]==3] 
y_train_letter_c[:,:] = 4
y_test_letter_c[:,:] = 4

x_train_letter_s =  x_train[y_train[:,0]==19] 
y_train_letter_s =  y_train[y_train[:,0]==19] 
x_test_letter_s =  x_test[y_test[:,0]==19] 
y_test_letter_s =  y_test[y_test[:,0]==19] 
y_train_letter_s[:,:] = 5
y_test_letter_s[:,:] = 5

x_train_letter_w =  x_train[y_train[:,0]==23] 
y_train_letter_w =  y_train[y_train[:,0]==23] 
x_test_letter_w =  x_test[y_test[:,0]==23] 
y_test_letter_w =  y_test[y_test[:,0]==23] 
y_train_letter_w[:,:] = 10
y_test_letter_w[:,:] = 10

#%%

# MNIST:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # remove 4,5. 2 swipes. 

x_train_no_45 =  x_train[ np.logical_and( y_train!=4, y_train!=5) ] 
y_train_no_45 =  y_train[ np.logical_and( y_train!=4, y_train!=5) ] 
x_test_no_45 =  x_test[ np.logical_and( y_test!=4, y_test!=5) ] 
y_test_no_45 =  y_test[ np.logical_and( y_test!=4, y_test!=5) ] 


#%%

# combine digts no 4,5 with letter c , s, w

x_train = np.vstack([x_train_no_45 , x_train_letter_c, x_train_letter_s, x_train_letter_w])
x_test = np.vstack([x_test_no_45 , x_test_letter_c, x_test_letter_s, x_test_letter_w])

y_train = np.hstack([y_train_no_45, y_train_letter_c.reshape([-1]), y_train_letter_s.reshape([-1]), y_train_letter_w.reshape([-1]) ])
y_test = np.hstack([y_test_no_45, y_test_letter_c.reshape([-1]), y_test_letter_s.reshape([-1]), y_test_letter_w.reshape([-1]) ])



#%%


for n in range(30000,30100):
    plt.imshow(x_train[n,:,:,0])
    plt.show()

#%%
# augment and manipulate the data, to make it more like our real data

def boolerize(X, threshold):
    X1 = X.copy()
    X1[X1>=threshold] = 255
    X1[X1<threshold] = 0
    return X1

thres = 1

x_train[x_train>=thres] = 255
x_train[x_train<thres] = 0

x_test[x_test>=thres] = 255
x_test[x_test<thres] = 0

x_train1 = x_train.copy()
x_test1 = x_test.copy()
for thres in [125,225,245,250]:
    x_train_new = boolerize(x_train, thres)
    x_train1 = np.append(x_train1, x_train_new,axis =0)
    x_test_new = boolerize(x_test,thres)
    x_test1 = np.append(x_test1, x_test_new,axis =0)
    

x_train = x_train1.copy()
x_test = x_test1.copy()
del x_train1, x_test1

y_train = np.hstack([y_train,y_train,y_train,y_train, y_train])
y_test = np.hstack([y_test,y_test,y_test,y_test, y_test])


#%%

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)





#%%

num_classes = 11
epochs = 50

try:
    model = load_model('my_model.h5')
except IOError:
    model = Sequential()
#    model.add(Conv2D(filters=8,kernel_size=5,activation='relu', input_shape=input_shape))
#    model.add(Dropout(0.5))
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    model.save('model_2_digits_letter_csw.h5')  # creates a HDF5 file 'my_model.h5'

score = model.evaluate(x_test, y_test, verbose=0)

y_pred = model.predict(x_test)
print(y_pred.shape)
l = y_pred.shape[1]
c_matrix = np.zeros((l,l))
for i in range(y_pred.shape[0]):
    # check the true label
    true_label = np.argmax(y_test[i])
    # check the prediction label
    pred_label = np.argmax(y_pred[i])
    c_matrix[true_label][pred_label] += 1
# print(y_pred)
# print(y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix.astype(int))




#%%



import numpy as np
import matplotlib.pyplot as plt

# 3:
a = [(118, 121), (118, 120), (118, 119), (118, 117), (119, 117), (120, 114), (122, 113), (128, 110), (133, 109), (150, 107), (155, 107), (170, 107), (174, 107), (180, 109), (187, 115), (191, 119), (195, 124), (195, 125), (195, 132), (195, 135), (195, 143), (195, 145), (194, 156), (193, 160), (189, 170), (188, 174), (186, 181), (185, 183), (182, 191), (178, 199), (176, 203), (174, 206), (171, 210), (170, 210), (167, 213), (162, 214), (159, 215), (158, 216), (158, 215), (160, 214), (162, 214), (167, 214), (168, 214), (172, 216), (174, 217), (179, 220), (180, 224), (181, 224), (184, 227), (186, 229), (189, 238), (192, 248), (193, 264), (193, 283), (193, 291), (192, 295), (189, 303), (189, 305), (187, 311), (187, 313), (184, 318), (183, 318), (181, 320), (176, 322), (171, 322), (170, 322), (162, 321), (158, 320), (150, 317), (149, 316), (146, 314), (145, 314), (144, 312), (138, 307), (134, 302), (130, 300), (129, 300), (128, 300), (127, 299), (126, 298), (126, 297)]

# 4:
#a = [(216, 229),(215, 229),(215, 229),(211, 230),(210, 230),(204, 230),(175, 229),(163, 228),(115, 225),(107, 225),(97, 224),(64, 219),(59, 218),(52, 217),(40, 217),(39, 217),(37, 216),(40, 212),(41, 211),(43, 209),(61, 194),(68, 189),(86, 170),(99, 150),(103, 147),(110, 138),(112, 134),(120, 121),(133, 107),(136, 102),(138, 100),(147, 91),(148, 89),(151, 85),(152, 85),(153, 84),(153, 83),(153, 102),(153, 113),(154, 143),(154, 153),(154, 188),(154, 200),(153, 258),(151, 297),(151, 307),(151, 361),(151, 329),(151, 350),(151, 358),(151, 360),(151, 360)]

n_points = len(a)

x = np.zeros((n_points,))
y = x.copy()

n=0
for a0 in a:
    x[n] = a0[1]
    y[n] = a0[0]
    n+=1
    
mat0 = np.zeros((480,320))

for n in range(x.shape[0]):
    mat0[x[n].astype(int)][y[n].astype(int)] = 1
    
#plt.imshow(mat0)
    

    
x += 80

dx = x.max() - x.min()
dy = y.max() - y.min()

ds = max(dx,dy)

margin = 60

ds += margin

for n_mul in range(1,18):
    if ds < n_mul*28:
        break

x_start = (x.min()+x.max())//2 - ds//2
y_start = (y.min()+y.max())//2 - ds//2

x_new = x - x_start
y_new = y - y_start

new_mat2 = np.zeros((28,28))

x_in_new_mat2 = x_new.astype(int) // n_mul
y_in_new_mat2 = y_new.astype(int) // n_mul

for n in range(n_points):
    new_mat2[x_in_new_mat2[n]][y_in_new_mat2[n]] = 1



#new_mat = np.zeros((n*28,n*28)) 
#for n in range(x.shape[0]):
#    new_mat[x_new[n].astype(int)][y_new[n].astype(int)] = 1
#    
##plt.imshow(new_mat)
#
#def shrink_image(original_image, threshold = 0.1):
#    '''
#    :param original_image: the original image with (28*n) x (28*n) dimension, a numpy array
#    :return: return the shrinked image 28 x 28, a numpy array
#    '''
#    n = original_image.shape[0]//28
#    x1=0
#    
#    image = np.zeros((28, 28))
#    while x1 < 28:
#        y1=0
#        while y1 < 28:
#
#            sample_square = original_image[x1*n: (x1 + 1)*n, y1*n: (y1 + 1)*n]
#            count = np.sum(sample_square.flatten())
#            if count > 0:
#                image[x1][y1] = 1
#
#            y1 += 1
#
#        x1 += 1
#
#    return image
#
#
#
#new_mat2 = shrink_image(new_mat,0.00001)

plt.imshow(new_mat2)



#%%

nn_input = new_mat2.reshape(1,28,28,1) 
Y = np.argmax(model.predict(nn_input))
print(Y)

#plt.imshow(nn_input[0,:,:,0])


#%%
n_points = len(a)

print('int x[%d] = {' % n_points)
for n in range(n_points):
    print(a[n][0],end = '')
    if n!= n_points-1:
        print(',',end='')
print('};')
    
print('int y[%d] = {' % n_points)
for n in range(n_points):
    print(a[n][1],end = '')
    if n!= n_points-1:
        print(',',end='')
print('};')
    




#%%
    
from keras import backend as K

nn_input = new_mat2.reshape(1,28,28,1) 

get_1st_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[0].output])
layer_output1 = get_1st_layer_output([nn_input,0])[0] 



# # compare for 4
compare = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
]).reshape([1,-1])

# # compare for 3
#compare = np.array([
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#        ]).reshape([1,-1])
    
np.abs(compare - layer_output1).sum()

#%%
get_1nd_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[1].output])
layer_output2 = get_1nd_layer_output([nn_input,0])[0] 


#%%

get_2nd_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[3].output])
layer_output3 = get_2nd_layer_output([nn_input,0])[0] 

#%%

def get_int(a):
    return (2**18*a).astype(int)

#%%


#    fc layer 1

n=0
for layer in model.layers:
    if layer.name.split('_')[0]=='dense' :
        n+=1
        if n==1:
            weights  = layer.get_weights()
            coefs = weights[0]
            bias = weights[1]    
    
filters = coefs.transpose()
print("const int dense1_coef[DENSE1_OUTPUT][DENSE1_INPUT] = {",end='')
for m in range(filters.shape[0]): 
    print('{',end='')
    for k in range(filters.shape[1]):
        print(get_int(filters[m,k]),end='')
        if k+1 < filters.shape[1]:
            print(",",end='')
        elif m+1<filters.shape[0]: 
            print("},",end ='')
        else: 
            print("}}",end ='')
print(";")
    

print("const int dense1_bias[DENSE1_OUTPUT] = {", end = '')
for n in range(bias.shape[0]):
    print(get_int(bias[n] *256 ),end='')
    if n+1 < bias.shape[0]:
        print(",",end='')
print("};")


#%% fc layer 2. the final layer.
n=0
for layer in model.layers:
    if layer.name.split('_')[0]=='dense' :
        n+=1
        if n==2:
            weights  = layer.get_weights()
            coefs = weights[0]
            bias = weights[1]    
    
    
filters = coefs.transpose()
print("const int dense2_coef[DENSE2_OUTPUT][DENSE2_INPUT] = {",end='')
for m in range(filters.shape[0]): 
    print('{',end='')
    for k in range(filters.shape[1]):
        print(get_int( filters[m,k] ),end='')
#        print(( filters[m,k] ),end='')
        if k+1 < filters.shape[1]:
            print(",",end='')
        elif m+1<filters.shape[0]: 
            print("},",end ='')
        else: 
            print("}}",end ='')
print(";")
    

print("const int dense2_bias[DENSE2_OUTPUT] = {", end = '')
for n in range(bias.shape[0]):
    print( get_int(bias[n])*256,end='')
#    print( (bias[n]),end='')
    if n+1 < bias.shape[0]:
        print(",",end='')
print("};")





