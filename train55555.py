import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import numpy as np
import os

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as gt:
        fg_bgs, fg_bg_softmaxes, bboxes = model.call(data)

        loss1 = rpn.rpn_object_loss(labels[0], fg_bg_softmaxes)
        loss2 = rpn.rpn_bbox_loss(labels[0], labels[1], bboxes)
        loss = loss1 + loss2

        grads = gt.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss1, loss2, loss

model = rpn.RPN(backbone.Resnet34_FPN(), 3)
weights_path = "weights.ckpt"
#model.load_weights(weights_path)

optimizer = tf.keras.optimizers.Adam(0.001)

ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 1)
#ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
anchors = anchor_utils.get_all_anchors((512, 512), [64, 128, 256, 512, 1024], [(1, 1), (1, 2), (2, 1)])

data1, data2, data3 = ds.next_batch()
data2, data3 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, data2)

for i in range(100):
    l1, l2, l = train_step(model, optimizer, [data1, data2, data3], [data2, data3])
    print(i, tf.keras.backend.eval(l1), tf.keras.backend.eval(l2), tf.keras.backend.eval(l))

    model.save_weights(weights_path)

