
import cv2
from PIL import Image
import numpy as np

import pyautogui as pag

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

# set Camera
video = cv2.VideoCapture(0)


video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print('width', video.get(cv2.CAP_PROP_FRAME_WIDTH))
print('height', video.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, img_raw = video.read()
    img = Image.fromarray(np.uint8(img_raw))
    # img = img.resize((224, 224))
    img = image.img_to_array(img)

    x1 = int((1280-224)/2)
    # y1 = int((720-224)/2)
    y1 = int(100)
    
    img = img[y1:y1+224, x1:x1+224, :]
    img = np.expand_dims(img, axis=0)
    img = preprocess_mobilenet(img)
    print('predicting')
    out = cnn_model.predict(x=img)
    pag.moveTo(out[0, 0], out[0, 1])
    print(out[0, 0], out[0, 1])

    cv2.rectangle(img_raw, (x1, y1), (x1+224, y1+224), (0, 255, 0), 5)
    cv2.imshow('hello', img_raw)
    if cv2.waitKey(1) & 0xff == 27:
        break
cv2.destroyAllWindows()
