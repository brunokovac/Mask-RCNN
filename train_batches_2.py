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
model.load_weights(weights_path)

ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 16)
#ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
anchors = anchor_utils.get_all_anchors((512, 512), [64, 128, 256, 512, 1024], [(1, 1), (1, 2), (2, 1)])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.00001, decay_steps=ds.total_batches * 2, decay_rate=0.95, staircase=True)
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=ds.total_batches * 2, decay_rate=0.9, staircase=True)
optimizer = tf.keras.optimizers.Adam(lr_schedule)

for epoch in range(10):
    print("Epoch", (epoch + 1))
    for i in range(ds.total_batches):
        data1, gt_boxes, data3, d4 = ds.next_batch()
        data2, data3 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, gt_boxes)
        l1, l2, l = train_step(model, optimizer, [data1, data2, data3], [data2, data3])
        print(i, "cls_loss: ", tf.keras.backend.eval(l1), "bbox_loss: ", tf.keras.backend.eval(l2), "total: ",tf.keras.backend.eval(l))

    model.save_weights(weights_path)

