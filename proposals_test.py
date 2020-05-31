import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.transform as skt
import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import numpy as np
import sys
import heatmap_utils
import mask_rcnn
import image_util
import losses

if __name__ == "__main__":
    tf.executing_eagerly()
    #train_dataset = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 2)
    train_dataset = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 5)
    #valid_dataset = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/valid_list.txt", 2)
    valid_dataset = dataset_util.Dataset("dataset/VOC2012", "/valid_list.txt", 5)
    anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

    backbone2 = backbone.Resnet34_FPN()
    rpn2 = rpn.RPN(backbone2, 3)
    model = mask_rcnn.Mask_RCNN(rpn2, anchors, len(config.CLASSES))

    checkpoint = tf.train.Checkpoint(net=model, step=tf.Variable(1))
    manager = tf.train.CheckpointManager(checkpoint, config.WEIGHTS_DIR, max_to_keep=4)
    if manager.latest_checkpoint:
        print("Restoring...", manager.latest_checkpoint)
        images, gt_boxes, gt_classes, img_sizes = train_dataset.next_batch()
        #model([images, img_sizes], training=False)
        checkpoint.restore(manager.latest_checkpoint)

    images, gt_boxes, gt_classes, img_sizes = train_dataset.next_batch()
    gt_rpn_classes, gt_rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)

    data = [images, img_sizes]
    _, _, rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=False)
    ind = np.where(gt_rpn_classes == 1)

    for i in range(len(images)):
        image_util.draw_bounding_boxes_from_array("train-proposals{}.png".format(i), images[i].astype(np.uint8), proposals[i][:5])

    for i in range(len(images)):
        gt_rpn_classes_i = gt_rpn_classes[i]
        deltas = rpn_bbox_deltas[i][gt_rpn_classes_i==1]
        anchors2 = anchors[gt_rpn_classes_i==1]

        height = anchors2[:, 3] - anchors2[:, 1]
        width = anchors2[:, 2] - anchors2[:, 0]
        center_y = anchors2[:, 1] + 0.5 * height
        center_x = anchors2[:, 0] + 0.5 * width

        center_y += deltas[:, 1] * height
        center_x += deltas[:, 0] * width
        height *= tf.exp(deltas[:, 3])
        width *= tf.exp(deltas[:, 2])

        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width

        boxes2 = tf.stack([x1, y1, x2, y2], axis=1)

        image_util.draw_bounding_boxes_from_array("deltas-train{}.png".format(i), images[i].astype(np.uint8), boxes2)

    images, gt_boxes, gt_classes, img_sizes = valid_dataset.next_batch()
    gt_rpn_classes, gt_rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)

    data = [images, img_sizes]
    _, _, rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=False)
    print(rpn_bbox_deltas.shape, proposals.shape)
    print(rpn_fg_bg_softmaxes[gt_rpn_classes == 1])

    for i in range(len(images)):
        image_util.draw_bounding_boxes_from_array("valid-proposals{}.png".format(i), images[i].astype(np.uint8), proposals[i][:5])

    for i in range(len(images)):
        gt_rpn_classes_i = gt_rpn_classes[i]
        deltas = rpn_bbox_deltas[i][gt_rpn_classes_i==1]
        anchors2 = anchors[gt_rpn_classes_i==1]

        height = anchors2[:, 3] - anchors2[:, 1]
        width = anchors2[:, 2] - anchors2[:, 0]
        center_y = anchors2[:, 1] + 0.5 * height
        center_x = anchors2[:, 0] + 0.5 * width

        center_y += deltas[:, 1] * height
        center_x += deltas[:, 0] * width
        height *= tf.exp(deltas[:, 3])
        width *= tf.exp(deltas[:, 2])

        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width

        boxes2 = tf.stack([x1, y1, x2, y2], axis=1)

        image_util.draw_bounding_boxes_from_array("deltas-valid{}.png".format(i), images[i].astype(np.uint8), boxes2)

