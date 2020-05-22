import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import mask_rcnn
import losses
import numpy as np
import sys

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as gt:
        gt_boxes, gt_classes, img_sizes, gt_rpn_classes, gt_rpn_bbox_deltas = labels

        rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=True)

        rpn_object_loss = losses.rpn_object_loss(gt_rpn_classes, rpn_fg_bg_softmaxes)
        rpn_bbox_loss = losses.rpn_bbox_loss(gt_rpn_classes, gt_rpn_bbox_deltas, rpn_bbox_deltas)

        mask_rcnn_gt_proposals, mask_rcnn_predicted_classes, mask_rcnn_predicted_bbox_deltas, \
        mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_gt_masks = \
            mask_rcnn.generate_mask_rcnn_labels(proposals, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, gt_classes, gt_boxes)
        mask_rcnn_class_loss = losses.mask_rcnn_class_loss(mask_rcnn_gt_classes, mask_rcnn_predicted_classes)
        mask_rcnn_bbox_loss = losses.mask_rcnn_bbox_loss(mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_predicted_bbox_deltas)

        loss = rpn_object_loss + rpn_bbox_loss + mask_rcnn_class_loss + mask_rcnn_bbox_loss
        #loss = rpn_object_loss + rpn_bbox_loss

    grads = gt.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rpn_object_loss, rpn_bbox_loss, mask_rcnn_class_loss, mask_rcnn_bbox_loss

def valid(model, dataset):
    print("*" * 10, "STARTING VALIDATION", "*" * 10)

    valid_losses = np.zeros(4)

    for _ in range(dataset.total_batches):
        images, gt_boxes, gt_classes, img_sizes = dataset.next_batch()
        gt_rpn_classes, gt_rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)

        data = [images, img_sizes]
        _, _, rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=False)

        rpn_object_loss = losses.rpn_object_loss(gt_rpn_classes, rpn_fg_bg_softmaxes)
        rpn_bbox_loss = losses.rpn_bbox_loss(gt_rpn_classes, gt_rpn_bbox_deltas, rpn_bbox_deltas)

        mask_rcnn_gt_proposals, mask_rcnn_predicted_classes, mask_rcnn_predicted_bbox_deltas, \
        mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_gt_masks = \
            mask_rcnn.generate_mask_rcnn_labels(proposals, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, gt_classes, gt_boxes)
        mask_rcnn_class_loss = losses.mask_rcnn_class_loss(mask_rcnn_gt_classes, mask_rcnn_predicted_classes)
        mask_rcnn_bbox_loss = losses.mask_rcnn_bbox_loss(mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_predicted_bbox_deltas)

        valid_losses += [rpn_object_loss, rpn_bbox_loss, mask_rcnn_class_loss, mask_rcnn_bbox_loss]

    valid_losses /= dataset.total_batches
    print("*" * 15, "VALID", "*" * 15)
    print("rpn_cls_loss={}, rpn_bbox_loss={}, mask_rcnn_cls_loss={}, mask_rcnn_bbox_loss={},".format(*valid_losses))
    print("*" * 40)
    return valid_losses

train_dataset = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 20)
#train_dataset = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 8)
valid_dataset = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/valid_list.txt", 20)
#valid_dataset = dataset_util.Dataset("dataset/VOC2012", "/valid_list.txt", 8)
#ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-4)

backbone = backbone.Resnet34_FPN()
rpn = rpn.RPN(backbone, 3)
model = mask_rcnn.Mask_RCNN(rpn, anchors, len(config.CLASSES))

checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model, step=tf.Variable(1))
manager = tf.train.CheckpointManager(checkpoint, config.WEIGHTS_DIR, max_to_keep=4)
if manager.latest_checkpoint:
    print("Restoring...", manager.latest_checkpoint)
    images, gt_boxes, gt_classes, img_sizes = train_dataset.next_batch()
    rpn_classes, rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)
    l1, l2, l3, l4 = train_step(model, optimizer, [images, img_sizes], [gt_boxes, gt_classes, img_sizes, rpn_classes, rpn_bbox_deltas])
    checkpoint.restore(manager.latest_checkpoint).assert_consumed()


if __name__ == "__main__":
    train_losses_file = open(config.TRAIN_LOSSES_FILE, "a+")
    valid_losses_file = open(config.VALID_LOSSES_FILE, "a+")

    max_loss = float("inf")
    bigger_loss_in_row = 0
    for epoch in range(1, 501):
        print("Epoch", epoch)
        epoch_losses = np.zeros(4)
        for i in range(1, train_dataset.total_batches + 1):
            images, gt_boxes, gt_classes, img_sizes = train_dataset.next_batch()
            rpn_classes, rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)
            l1, l2, l3, l4 = train_step(model, optimizer, [images, img_sizes], [gt_boxes, gt_classes, img_sizes, rpn_classes, rpn_bbox_deltas])

            l1, l2, l3, l4 = tf.keras.backend.eval(l1), tf.keras.backend.eval(l2), tf.keras.backend.eval(l3), tf.keras.backend.eval(l4)
            epoch_losses += [l1, l2, l3, l4]

            if i % 10 == 0:
                print("Iter {}: rpn_cls_loss={}, rpn_bbox_loss={}, mask_rcnn_cls_loss={}, mask_rcnn_bbox_loss={},".format(i, l1, l2, l3, l4))

        epoch_losses /= train_dataset.total_batches
        print("*" * 50)
        print("Epoch {}: rpn_cls_loss={}, rpn_bbox_loss={}, mask_rcnn_cls_loss={}, mask_rcnn_bbox_loss={},".format(epoch, *epoch_losses))
        print("*" * 50)
        with open(config.TRAIN_LOSSES_FILE, "a+") as f1:
            f1.write("{} {} {} {}\n".format(*epoch_losses))

        if epoch % 4 == 0:
            checkpoint.step.assign_add(1)
            manager.save()

            valid_losses = valid(model, valid_dataset)
            valid_loss = np.sum(valid_losses)

            with open(config.VALID_LOSSES_FILE, "a+") as f2:
                f2.write("{} {} {} {}\n".format(*valid_losses))

            if valid_loss < max_loss:
                max_loss = valid_loss
                bigger_loss_in_row = 0
                checkpoint.step.assign_add(1)
                manager.save()
            else:
                bigger_loss_in_row += 1

                if bigger_loss_in_row == 100:
                    print("{}. bigger loss in row, exiting".format(bigger_loss_in_row))
                    sys.exit(0)



