import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import mask_rcnn
import losses
import numpy as np

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as gt:
        gt_boxes, gt_classes, img_sizes, gt_rpn_classes, gt_rpn_bbox_deltas = labels

        rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=True)

        rpn_object_loss = losses.rpn_object_loss(rpn_classes, rpn_fg_bg_softmaxes)
        rpn_bbox_loss = losses.rpn_bbox_loss(rpn_classes, gt_rpn_bbox_deltas, rpn_bbox_deltas)

        mask_rcnn_gt_proposals, mask_rcnn_predicted_classes, mask_rcnn_predicted_bbox_deltas, \
        mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_gt_masks = \
            mask_rcnn.generate_mask_rcnn_labels(proposals, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, gt_classes, gt_boxes)
        mask_rcnn_class_loss = losses.mask_rcnn_class_loss(mask_rcnn_gt_classes, mask_rcnn_predicted_classes)
        mask_rcnn_bbox_loss = losses.mask_rcnn_bbox_loss(mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_predicted_bbox_deltas)

        loss = rpn_object_loss + rpn_bbox_loss + mask_rcnn_class_loss + mask_rcnn_bbox_loss

    grads = gt.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rpn_object_loss, rpn_bbox_loss, mask_rcnn_class_loss, mask_rcnn_bbox_loss

#ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 16)
ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 16)
anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.02, decay_steps=ds.total_batches * 10,
                                                             decay_rate=0.90, staircase=True)
#optimizer = tf.keras.optimizers.SGD(lr_schedule, nesterov=True)
optimizer = tf.keras.optimizers.SGD(lr=0.002, momentum=0.9, decay=1e-4, nesterov=True)

backbone = backbone.Resnet34_FPN()
rpn = rpn.RPN(backbone, 3)
model = mask_rcnn.Mask_RCNN(rpn, anchors, len(config.CLASSES))

#weights_path = config.WEIGHTS_PATH
model([np.random.rand(1, 512, 512, 3), np.array([[512, 512]])], training=True)
#model.load_weights(weights_path)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model, step=tf.Variable(1))
manager = tf.train.CheckpointManager(checkpoint, config.WEIGHTS_DIR, max_to_keep=4)
if manager.latest_checkpoint:
    print("Restoring...", manager.latest_checkpoint)
    checkpoint.restore(manager.latest_checkpoint)

for epoch in range(1, 10):
    print("Epoch", epoch)
    for i in range(1, ds.total_batches + 1):
        images, gt_boxes, gt_classes, img_sizes = ds.next_batch()
        rpn_classes, rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)
        l1, l2, l3, l4 = train_step(model, optimizer, [images, img_sizes], [gt_boxes, gt_classes, img_sizes, rpn_classes, rpn_bbox_deltas])

        print("Iter {}: rpn_cls_loss={}, rpn_bbox_loss={}, mask_rcnn_cls_loss={}, mask_rcnn_bbox_loss={},".format(
            i, tf.keras.backend.eval(l1), tf.keras.backend.eval(l2), tf.keras.backend.eval(l3), tf.keras.backend.eval(l4)))

    if epoch % 3 == 0:
        checkpoint.step.assign_add(1)
        manager.save()

