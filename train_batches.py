import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import os

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as gt:
        model(data)
        fg_bgs, fg_bg_softmaxes, bboxes = model(data, training=True)

        loss1 = rpn.rpn_object_loss(labels[0], fg_bg_softmaxes)
        loss2 = rpn.rpn_bbox_loss(labels[0], labels[1], bboxes)
        loss = loss1 + loss2

        grads = gt.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss1, loss2, loss

ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 32)
#ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=ds.total_batches * 5,
                                                             decay_rate=0.90, staircase=True)
#optimizer = tf.keras.optimizers.SGD(lr_schedule, nesterov=True)
optimizer = tf.keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=1e-4, nesterov=True)

backbone = backbone.Resnet34_FPN()
backbone.compile(optimizer)
model = rpn.RPN(backbone, 3)
model.compile(optimizer)
weights_path = "weights_all.ckpt"
model.load_weights(weights_path)

for epoch in range(50):
    print("Epoch", (epoch + 1))
    for i in range(ds.total_batches):
        data1, gt_boxes, data3, d4 = ds.next_batch()
        data2, data3 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, gt_boxes)
        l1, l2, l = train_step(model, optimizer, [data1, data2, data3], [data2, data3])
        print("Iter {}: rpn_cls_loss={}, rpn_bbox_loss={}, rpn_total_loss={}".format(i+1,
                    tf.keras.backend.eval(l1), tf.keras.backend.eval(l2), tf.keras.backend.eval(l)))

    model.save_weights(weights_path)

