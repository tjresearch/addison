import sys
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
from train_model_ import ResNet50

img_name = "cleaned_imgs_v4/000000.jpg"


img = cv2.imread(img_name)
img_float = cv2.resize(img, (224,224)).astype(np.float32)
img_float -= 128

img_in = np.expand_dims(img_float, axis=0)

model = ResNet50(include_top=False, load_weight=True, weights='models/run1_0.01_weights.12-3.80.hdf5',
                input_shape=(224,224,3))
pred = model.predict(img_in)

bboxes = utils.get_boxes(pred[0], cutoff=0.1)
bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.05)
draw = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=3, draw_dot=True, radius=3)
draw = draw.astype(np.uint8)

plt.imshow(draw[...,::-1])
plt.show()
