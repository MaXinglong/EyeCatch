#!usr/bin/env python3

import os
from keras.preprocessing import image
from PIL import Image
import cv2
import numpy as np
import json

root_dirs = os.listdir('./data')

# x
for root in root_dirs:
    fns = os.listdir('./data/' + root + '/')
    fns = [x for x in fns if x.split('.')[-1] == 'jpg']

    data = np.zeros((len(fns), 224, 224, 3))
    for i, fn in enumerate(fns):
        filename = './data/' + root + '/' + fn
        if filename.split('.')[-1] != 'jpg':
            continue
        img = image.load_img(filename, target_size=(224, 224))
        # type(x) float32
        x = image.img_to_array(img)

        data[i, :] = x

        print('%.2f%%' %(i/len(fns)))
    print('done...saving...waiting')
    np.save('./data/' + root + '/' + root + '-x.npy', data)

# y
for root in root_dirs:
    fn = './data/' + root + '/labels.json'
    js = json.load(open(fn))

    fns = os.listdir('./data/' + root + '/')
    fns = [x for x in fns if x.split('.')[-1] == 'jpg']
    
    labels = np.zeros((len(fns), 2))
    for i, fn in enumerate(fns):
        labels[i, :] = js['./data/' + root + '/' + fn]
        print('%.2f%%' %(i/len(fns)))

    print('done...saving...waiting')
    np.save('./data/' + root + '/' + root + '-y.npy', labels)
