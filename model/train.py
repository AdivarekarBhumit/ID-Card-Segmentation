import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint
from .unet_model import get_model

X = np.load('final_train.npy')
Y = np.load('final_mask.npy')

def t_generator(X_train, Y_train, batch_size):
  features = np.zeros(shape=(batch_size, 256,256,3))
  labels = np.zeros(shape=(batch_size, 256,256,1))
  while True:
    start = 0
    end = batch_size
    for i in range(243):
      features = X_train[start:end]
      labels = Y_train[start:end]
      start = end
      end = end + batch_size
      yield features / 255.0, labels / 255.0
      
def v_generator(X_val, Y_val, batch_size):
  features = np.zeros(shape=(batch_size, 256,256,3))
  labels = np.zeros(shape=(batch_size, 256,256,1))
  while True:
    start = 0
    end = batch_size
    for i in range(27):
      features = X_val[start:end]
      labels = Y_val[start:end]
      start = end
      end = end + batch_size
      yield features / 255.0, labels / 255.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    shuffle=True, 
                                                    random_state=265, 
                                                    test_size=0.1)
del X, Y
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                                  shuffle=True, 
                                                  random_state=265,
                                                  test_size=0.1)

model = get_model()

## Adding Callbacks
tb = TensorBoard(log_dir='./logs/', batch_size=8, write_graph=True)


model_chkpt = ModelCheckpoint('unet_model_whole_100epochs.h5',
                              monitor='val_loss', verbose=1, 
                              save_best_only=True)


history = model.fit_generator(generator = t_generator(X_train, Y_train, 50), 
                              steps_per_epoch=X_train.shape[0] // 50, 
                              epochs=100, callbacks=[tb, model_chkpt], 
                              validation_data=v_generator(X_val, Y_val, 50),
                              validation_steps=X_val.shape[0] // 50)

