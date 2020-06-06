import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import mask_rcnn
import numpy as np
import image_util
import metrics

#np.random.seed(101)
ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 2)
#ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 2)
anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

backbone = backbone.Resnet34_FPN()
rpn = rpn.RPN(backbone, 3)
model = mask_rcnn.Mask_RCNN(rpn, anchors, len(config.CLASSES))

checkpoint = tf.train.Checkpoint(net=model, step=tf.Variable(1))
manager = tf.train.CheckpointManager(checkpoint, config.WEIGHTS_DIR, max_to_keep=4)
if manager.latest_checkpoint:
    print("Restoring...", manager.latest_checkpoint)
    model([np.random.rand(1, 512, 512, 3), np.array([[512, 512]])], training=False)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

images, gt_boxes, gt_classes, gt_masks, img_sizes = ds.next_batch()
predicted_bboxes, classes_scores, classes, predicted_masks, fg_bg_softmaxes, rpn_bbox_deltas, \
    classes_softmax, bbox_deltas, pred_masks, proposals = model([images, img_sizes], training=False)

sum = 0
for i in range(len(images)):
    '''ap, _, _, _ = metrics.compute_ap(gt_boxes[i], gt_classes[i], gt_masks,
                                     tf.keras.backend.eval(predicted_bboxes[i]),
                                     tf.keras.backend.eval(classes[i]),
                                     tf.keras.backend.eval(classes_scores[i]), predicted_masks[i])
    sum += ap'''

    boxes = predicted_bboxes[i]
    masks = predicted_masks[i]

    texts = []
    for j in range(len(boxes)):
        texts.append("{} {:.2f}".format(config.CLASSES[classes[i][j]].upper(), classes_scores[i][j]))

    image = images[i][:img_sizes[i][0], :img_sizes[i][1]]
    #image_util.draw_bounding_boxes_and_masks_from_array("img-train-{}".format(i), image, boxes, masks, texts)
    image_util.draw_bounding_boxes_from_array("img-train-{}.png".format(i), image, boxes, texts)
print(sum / len(images))
import sys; sys.exit(0)
ds = dataset_util.Dataset("dataset/VOC2012", "/valid_list.txt", 2)
images, gt_boxes, gt_classes, gt_masks, img_sizes = ds.next_batch()
predicted_bboxes, classes_scores, classes, predicted_masks, fg_bg_softmaxes, rpn_bbox_deltas, \
    classes_softmax, bbox_deltas, pred_masks, proposals = model([images, img_sizes], training=False)

sum = 0
for i in range(len(images)):
    ap, _, _, _ = metrics.compute_ap(gt_boxes[i], gt_classes[i], gt_masks,
                                     tf.keras.backend.eval(predicted_bboxes[i]),
                                     tf.keras.backend.eval(classes[i]),
                                     tf.keras.backend.eval(classes_scores[i]),
                                     tf.keras.backend.eval(predicted_masks[i]))
    sum += ap

    boxes = predicted_bboxes[i]
    masks = predicted_masks[i]

    texts = []
    for j in range(len(boxes)):
        texts.append("{} {:.2f}".format(config.CLASSES[classes[i][j]].upper(), classes_scores[i][j]))

    image = images[i][:img_sizes[i][0], :img_sizes[i][1]]
    image_util.draw_bounding_boxes_and_masks_from_array("img-train-{}".format(i), image, boxes, masks, texts)
print(sum / len(images))







