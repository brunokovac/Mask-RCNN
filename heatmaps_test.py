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

def draw_heatmaps(name, image, softmax):
    dim_shape = int(np.sqrt(len(softmax) / 3))
    shape = (dim_shape, dim_shape)

    plt.imshow(image/255)
    a = np.reshape(softmax[0::3], shape)
    a = skt.resize(a, (512, 512))
    plt.imshow(a, cmap='Reds', interpolation='nearest', alpha=0.4)
    plt.savefig(name + "-1.png")

    plt.imshow(image/255)
    a = np.reshape(softmax[1::3], shape)
    a = skt.resize(a, (512, 512))
    plt.imshow(a, cmap='Greens', interpolation='nearest', alpha=0.4)
    plt.savefig(name + "-2.png")

    plt.imshow(image/255)
    a = np.reshape(softmax[2::3], shape)
    a = skt.resize(a, (512, 512))
    plt.imshow(a, cmap='Blues', interpolation='nearest', alpha=0.4)
    plt.savefig(name + "-3.png")

if __name__ == "__main__":
    #tf.executing_eagerly()
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
        images, gt_boxes, gt_classes, gt_masks, img_sizes = valid_dataset.next_batch()
        model([images, img_sizes], training=False)
        checkpoint.restore(manager.latest_checkpoint)

    images, gt_boxes, gt_classes, gt_masks, img_sizes = train_dataset.next_batch()
    gt_rpn_classes, gt_rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)

    for i in range(len(images)):
        img = images[i]
        gt_rpn_classes_i = gt_rpn_classes[i]
        import image_util
        image_util.draw_bounding_boxes_from_array("anchors-sa2-{}.png".format(i), img.astype(np.uint8), anchors[gt_rpn_classes_i == 1])

    data = [images, img_sizes]
    _, _, _, rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=False)

    for j in range(len(images)):
        start = 0
        num_on_level = 128 * 128 * 3
        for k in range(2, 6):
            draw_heatmaps("TRAIN/img{}-level{}".format(j, k), images[j], rpn_fg_bg_softmaxes[j, start:start + num_on_level, 1])
            start += num_on_level
            num_on_level = num_on_level // 4

    print("kraj")
    import sys; sys.exit(0)

    images, gt_boxes, gt_classes, gt_masks, img_sizes = valid_dataset.next_batch()
    gt_rpn_classes, gt_rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)

    data = [images, img_sizes]
    _, _, _, rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, proposals = model(data, training=False)

    for j in range(len(images)):
        start = 0
        num_on_level = 128 * 128 * 3
        for k in range(2, 6):
            draw_heatmaps("VALID/img{}-level{}".format(j, k), images[j], rpn_fg_bg_softmaxes[j, start:start + num_on_level, 1])
            start += num_on_level
            num_on_level = num_on_level // 4


