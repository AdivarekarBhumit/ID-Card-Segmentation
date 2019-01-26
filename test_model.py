import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from model.iou_loss import IoU

model = load_model('unet_model_whole_100epochs.h5', compile=False)
model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

image_name = 'test.tif'

img = cv2.imread(image_name)
img = cv2.resize(img, (256,256))
img = img / 255.0
predict = model.predict(img.reshape(1,256,256,3))

output = predict[0].astype(np.int32)

cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()