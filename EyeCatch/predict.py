
import cv2
from PIL import Image
import numpy as np

import pyautogui as pag

# set Camera
video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
print('width', video.get(cv2.CAP_PROP_FRAME_WIDTH))
print('height', video.get(cv2.CAP_PROP_FRAME_HEIGHT))

from keras.models import load_model
from model.model import preprocess_mobilenet
from keras.preprocessing import image

from keras.applications import mobilenet


print('loading')
# get model
cnn_model = load_model('mobilenet_finetuning.h5',
                    custom_objects={
                        'relu6': mobilenet.relu6,
                        'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
print('loading done')

while True:
    ret, img_raw = video.read()
    img = Image.fromarray(np.uint8(img_raw))
    # img = img.resize((224, 224))
    img = image.img_to_array(img)

    x1 = int((1280-224)/2)
    y1 = int((720-224)/2)
    
    img = img[x1:x1+224, y1:y1+224]
    img = np.expand_dims(img, axis=0)
    img = preprocess_mobilenet(img)
    print('predicting')
    out = cnn_model.predict(x=img)
    pag.moveTo(out[0, 0], out[0, 1])
    print(out[0, 0], out[0, 1])

    cv2.imshow('hello', img_raw)
    if cv2.waitKey(300) & 0xff == 27:
        break
cv2.destroyAllWindows()
