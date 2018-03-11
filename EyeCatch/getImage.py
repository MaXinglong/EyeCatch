#!usr/bin/env python

import datetime
import os
import json

import cv2
import pyautogui as pag



video = cv2.VideoCapture(0)


video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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
    cv2.imshow('frame', image)
    x, y = pag.position()
    # pag.moveTo(100, 200)

    if get_flag == True:
        print('saving image')
        Id = Id + 1
        filename = '../data/' + rootname + '/D' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + 'D' + str(Id) + '.jpg'
        cv2.imwrite(filename, image)
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
