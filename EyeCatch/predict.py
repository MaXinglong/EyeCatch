from keras.models import load_model
from model.model import preprocess
from keras.preprocessing import image

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

video.set(cv2.CAP_PROP_CONTRAST, 0.5)
print('contrast', video.get(cv2.CAP_PROP_CONTRAST))

# get model
cnn_model = load_model('resnet50_finetuning.h5')


while True:
    ret, img = video.read()
    img = Image.fromarray(np.uint8(img))
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess(img)
    
    out = cnn_model.predict(x=img)
    # pag.moveTo(out[0, 0], out[0, 1])
    print(out[0, 0], out[0, 1])
    
    if cv2.waitKey(1) & 0xff == 27:
        break
