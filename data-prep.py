import numpy as np
import os
import cv2

data = os.listdir('train')

X = []
y = []

for i in range(len(data)):
    for j in range(len(os.listdir('train/' + data[i]))):
        img = np.load('train/' + data[i] + '/' +
                      os.listdir('train/' + data[i])[j])[0]
        img = cv2.resize(img, (32, 32))
        #img = np.stack([img, img, img], -1)
        X.append(img)
        y.append(i)
    print(data[i] + ' loaded')


X = np.array(X)
y = np.array(y)

np.save('train_X.npy', X)
np.save('train_y.npy', y)


val = os.listdir('val')

val_X = []
val_y = []

for i in range(len(val)):
    for j in range(len(os.listdir('val/' + val[i]))):
        img = np.load('val/' + val[i] + '/' +
                      os.listdir('val/' + val[i])[j])[0]
        img = cv2.resize(img, (32, 32))
        #img = np.stack([img, img, img], -1)
        val_X.append(img)
        val_y.append(i)
    print(val[i] + ' loaded')

val_X = np.array(val_X)
val_y = np.array(val_y)

np.save('test_X.npy', val_X)
np.save('test_y.npy', val_y)
