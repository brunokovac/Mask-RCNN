import rpn
import losses
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import os

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as gt:
        fg_bgs, fg_bg_softmaxes, bboxes, _ = model(data, training=True)

        object_loss = losses.rpn_object_loss(labels[0], fg_bg_softmaxes)
        bbox_loss = losses.rpn_bbox_loss(labels[0], labels[1], bboxes)
        loss = object_loss + bbox_loss

        grads = gt.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return object_loss, bbox_loss

ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 32)
#ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=ds.total_batches * 5,
                                                             decay_rate=0.90, staircase=True)
#optimizer = tf.keras.optimizers.SGD(lr_schedule, nesterov=True)
optimizer = tf.keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=1e-4, nesterov=True)

backbone = backbone.Resnet34_FPN()
model = rpn.RPN(backbone, 3)
weights_path = config.WEIGHTS_PATH
model.load_weights(weights_path)

for epoch in range(50):
    print("Epoch", (epoch + 1))
    for i in range(ds.total_batches):
        images, gt_boxes, gt_classes, img_sizes = ds.next_batch()
        rpn_classes, rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)
        object_loss, bbox_loss = train_step(model, optimizer, [images], [rpn_classes, rpn_bbox_deltas])
        print("Iter {}: rpn_cls_loss={}, rpn_bbox_loss={}".format(i+1, tf.keras.backend.eval(object_loss), tf.keras.backend.eval(bbox_loss)))

    model.save_weights(weights_path)

