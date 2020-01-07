import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import numpy as np

def one_hot(a):
    a = np.array(a, dtype=np.int32)
    b = np.zeros((a.size, a.max() + 1), dtype="float32")
    b[np.arange(a.size), a] = 1
    return b

model = rpn.RPN(backbone.Resnet34_FPN(), 3).model()
model.compile(opt=tf.keras.optimizers.Adam(0.001), metrics=["accuracy"])

images, bboxes, classes = dataset_util.Dataset("DATASET/VOC2012/VOC2012", 250).next_batch()

anchors = anchor_utils.get_all_anchors((512, 512), [32, 64, 128, 256, 512], [(1, 1), (1, 2), (2, 1)])
classes, bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, bboxes)

N = int(len(images) * 0.8)
images_train = images[:N]
bbox_deltas_train = bbox_deltas[:N]
classes_train = classes[:N]

images_val = images[N:]
bbox_deltas_val = bbox_deltas[N:]
classes_val = classes[N:]

model.fit([images_train, classes_train, bbox_deltas_train], [one_hot(classes_train), bbox_deltas_train],
          batch_size=64, epochs=2,
          validation_data=([images_val, classes_val, bbox_deltas_val], [one_hot(classes_val), bbox_deltas_val])
          )

