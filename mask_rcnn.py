import tensorflow as tf
import numpy as np
from backbone import *
import rpn
import anchor_utils
import dataset_util
import config

class Mask_RCNN(tf.keras.models.Model):

    def __init__(self, rpn, anchors):
        super().__init__()

        self.rpn = rpn
        self.anchors = anchors
        return

    def call(self, inputs, training=None):
        x = inputs[0]
        img_sizes = inputs[1]

        fg_bgs, fg_bg_softmaxes, bbox_deltas = self.rpn.call(inputs)

        ##### PROPOSALS #####
        fg_scores = fg_bg_softmaxes[:, :, 1]
        top_n = config.TEST_PRE_NMS_TOP_N if training else config.TEST_PRE_NMS_TOP_N
        _, indices = tf.math.top_k(fg_scores, top_n)

        boxes = []

        for i in range(indices.shape[0]):
            indices_i = indices[i]
            fg_scores_i = fg_scores[i]
            bbox_deltas_i = tf.gather(bbox_deltas[i], indices_i)
            anchors_i = tf.gather(self.anchors, indices_i)
            anchors_i = tf.cast(anchors_i, "float32")

            height = anchors_i[:, 3] - anchors_i[:, 1]
            width = anchors_i[:, 2] - anchors_i[:, 0]
            center_y = anchors_i[:, 1] + 0.5 * height
            center_x = anchors_i[:, 0] + 0.5 * width

            center_y += bbox_deltas_i[:, 1] * height
            center_x += bbox_deltas_i[:, 0] * width
            height *= tf.exp(bbox_deltas_i[:, 3])
            width *= tf.exp(bbox_deltas_i[:, 2])

            y1 = center_y - 0.5 * height
            x1 = center_x - 0.5 * width
            y2 = y1 + height
            x2 = x1 + width

            boxes_i = tf.stack([x1, y1, x2, y2], axis=1)

            height = tf.cast(img_sizes[i, 0], "float32")
            width = tf.cast(img_sizes[i, 1], "float32")
            correct_boxes_indices = tf.where(tf.greater_equal(boxes_i[:, 0], 0))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.less(boxes_i[:, 2], width))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.greater_equal(boxes_i[:, 1], 0))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.less(boxes_i[:, 3], height))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)

            nms_scores = tf.gather_nd(fg_scores_i, correct_boxes_indices)

            if training:
                num_rois = config.TRAIN_POST_NMS_ROIS
                nms_indices = tf.image.non_max_suppression(boxes_i, nms_scores, num_rois)
            else:
                num_rois = config.TEST_POST_NMS_ROIS
                nms_indices = tf.image.non_max_suppression(boxes_i, nms_scores, num_rois)

            boxes_i = tf.gather(boxes_i, nms_indices)
            print(boxes_i.shape)
            boxes.append(boxes_i)

        #boxes = tf.convert_to_tensor(boxes)
        ##########################################################################

        return boxes

if __name__ == "__main__":
    anchors = anchor_utils.get_all_anchors((512, 512), [64, 128, 256, 512, 1024], [(1, 1), (1, 2), (2, 1)])
    rpn_model = rpn.RPN(Resnet34_FPN(), 3)
    #weights_path = "weights.ckpt"
    #rpn_model.load_weights(weights_path)
    model = Mask_RCNN(rpn_model, anchors)

    #ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 2)
    ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
    data1, data2, data3, d4 = ds.next_batch()
    data2, data3 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, data2)

    res = model.call([data1, d4])

    import image_util
    for i in range(len(data1)):
        r = res[i]
        texts = []
        print(r)
        for j in range(len(r)):
            texts.append("t{}".format(j+1))
        image_util.draw_bounding_boxes_from_array(data1[i].astype(np.uint8), r, texts)