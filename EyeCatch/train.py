import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from getData import get_data
from model.model import get_model
from model.model import preprocess

import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
KTF.set_session(tf.Session(config=config))

cnn_model = get_model()
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.summary()

np.random.seed(1)
X, Y = get_data()
m = np.shape(X)[0]

choice = np.random.permutation(m)

X = X[choice]
Y = Y[choice]

X = preprocess(X)

print('fitting...')
cnn_model.fit(x=X, y=Y, batch_size=32, epochs=50, validation_split=0.2, shuffle=True)
cnn_model.save('resnet50_finetuning.h5')
print('done')
