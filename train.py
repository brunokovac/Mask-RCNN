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

images, bboxes, classes = dataset_util.Dataset("dataset/VOC2012", 2).next_batch()

anchors = anchor_utils.get_all_anchors((512, 512), [32, 64, 128, 256, 512], [(1, 1), (1, 2), (2, 1)])
classes, bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(2, anchors, bboxes)

model.fit([images, classes, bbox_deltas], [one_hot(classes), bbox_deltas])

