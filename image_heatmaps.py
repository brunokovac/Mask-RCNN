import tensorflow as tf
import dataset_util
import numpy as np

model = tf.keras.applications.ResNet50(include_top=True, weights="imagenet")
print(model.summary())

ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 2)
images, gt_boxes, gt_classes, gt_masks, img_sizes = ds.next_batch()

print(np.argmax(model(images), axis=1))
import sys
sys.exit(0)

last_layer = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer('conv4_block6_out').output, model.get_layer('conv4_block5_out').output])
res = last_layer(images, training=True)
print(res)