#!usr/bin/env python

import datetime
import os
import json

import cv2
import pyautogui as pag



video = cv2.VideoCapture(0)


video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print('width', video.get(cv2.CAP_PROP_FRAME_WIDTH))
print('height', video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# video.set(cv2.CAP_PROP_CONTRAST, 0.5)
# print('contrast', video.get(cv2.CAP_PROP_CONTRAST))


get_flag = False

labels = {}

rootname = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.mkdir('../data/'+rootname)

Id = 0

ret, image = video.read()
while ret:    

    x1 = int((1280-224)/2)
    y1 = int((720-224)/2)

    image_copy = image.copy()
    cv2.rectangle(image_copy, (x1, y1), (x1+224, y1+224), (0, 255, 0), 5)

    cv2.imshow('frame', image_copy)
    x, y = pag.position()
    # pag.moveTo(100, 200)

    if get_flag == True:
        print('saving image')
        Id = Id + 1
        filename = '../data/' + rootname + '/D' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + 'D' + str(Id) + '.jpg'
        
        image_save = image[y1:y1+224, x1:x1+224]
        cv2.imwrite(filename, image_save)
        labels[filename] = [x, y]
    else:
        print('waiting for save')

    key = cv2.waitKey(2)
    if key & 0xff == ord('b'):      #begin
        get_flag = True
    elif key & 0xff == ord('s'):    #stop
        get_flag = False
    elif key & 0xff == 27:          #esc
        cv2.destroyAllWindows()
        break

    ret, image = video.read()

if len(labels) != 0:
    filename = '../data/' + rootname + '/labels.json'
    with open(filename, 'w') as fp:
        json.dump(labels, fp)
