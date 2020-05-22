import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import mask_rcnn
import numpy as np
import image_util

tf.executing_eagerly()
#ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 5)
ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 2)
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

for _ in range(ds.total_batches):
    images, gt_boxes, gt_classes, img_sizes = ds.next_batch()

    predicted_bboxes, classes_scores, fg_bg_softmaxes, rpn_bbox_deltas, classes_softmax, bbox_deltas, proposals = model([images, img_sizes], training=False)

    for i in range(len(images)):
        boxes = []
        texts = []
        for j in range(len(config.CLASSES) - 1):
            for k in range(len(predicted_bboxes[i][j])):
                boxes.append(predicted_bboxes[i][j][k])
                texts.append("{} {:.2f}".format(config.CLASSES[j + 1], classes_scores[i][j][k]))

        image = images[i][:img_sizes[i][0], :img_sizes[i][1]]
        image_util.draw_bounding_boxes_from_array("img-test{}.png".format(i), image.astype(np.uint8), boxes, texts)

    import sys
    sys.exit(0)







