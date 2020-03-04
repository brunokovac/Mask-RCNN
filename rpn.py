import tensorflow as tf
import numpy as np
from backbone import *
import config

def rpn_object_loss(object_gt, object_predicted):
    object_gt = tf.squeeze(object_gt, axis=-1)
    indices = tf.where(tf.not_equal(object_gt, 0))

    gt_selected = tf.gather_nd(object_gt, indices)
    gt_selected = tf.cast(tf.equal(gt_selected, 1), tf.int32)
    pred_selected = tf.gather_nd(object_predicted, indices)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_selected, pred_selected)
    return loss

def rpn_bbox_loss(object_gt, bbox_gt, bbox_predicted):
    object_gt = tf.squeeze(object_gt, axis=-1)
    indices = tf.where(tf.equal(object_gt, 1))

    gt_bbox_selected = tf.gather_nd(bbox_gt, indices)
    pred_bbox_selected = tf.gather_nd(bbox_predicted, indices)

    diff = tf.abs(tf.cast(gt_bbox_selected, "float32") - pred_bbox_selected)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_mean(loss)
    return loss

class RPN(tf.keras.models.Model):

    def __init__(self, backbone, num_anchors):
        super().__init__()

        self.backbone = backbone
        self.num_anchors = num_anchors

        self.conv = tf.keras.layers.Conv2D(256, (3, 3), padding="same")
        self.relu = tf.keras.layers.Activation("relu")

        self.fg_bg_conv = tf.keras.layers.Conv2D(self.num_anchors * 2, (1, 1))
        self.fg_bg_reshape = tf.keras.layers.Lambda(lambda x : tf.reshape(x, [tf.shape(x)[0], -1, 2]), name="rpn_fg_bg")

        self.fg_bg_softmax = tf.keras.layers.Activation("softmax", name="rpn_fg_bg_softmax")

        self.bbox_conv = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1))
        self.bbox_reshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]), name="rpn_bbox")

        return

    def call(self, inputs, training=None):
        x = inputs[0]

        fg_bg_softmaxes = []
        fg_bgs = []
        bbox_deltas = []

        for level in self.backbone.call(x):
            y = self.conv(level)
            y = self.relu(y)

            fg_bg_conv = self.fg_bg_conv(y)
            #fg_bg_conv = tf.transpose(fg_bg_conv, [0, 3, 1, 2])
            fg_bg = self.fg_bg_reshape(fg_bg_conv)
            fg_bgs.append(fg_bg)

            fg_bg_softmax = self.fg_bg_softmax(fg_bg)
            fg_bg_softmaxes.append(fg_bg_softmax)

            bbox_conv = self.bbox_conv(y)
            #bbox_conv = tf.transpose(bbox_conv, [0, 3, 1, 2])
            bbox = self.bbox_reshape(bbox_conv)
            bbox_deltas.append(bbox)

        fg_bgs = tf.concat(fg_bgs, axis=1)
        fg_bg_softmaxes = tf.concat(fg_bg_softmaxes, axis=1)
        bbox_deltas = tf.concat(bbox_deltas, axis=1)

        return fg_bgs, fg_bg_softmaxes, bbox_deltas

    def model(self):
        inputs = [tf.keras.layers.Input(shape=(512, 512, 3)),
                  tf.keras.layers.Input(shape=(None, 1)),
                  tf.keras.layers.Input(shape=(None, 4))]
        outputs = self.call(inputs)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        model.add_loss(rpn_object_loss(inputs[1], outputs[0]))
        model.add_loss(rpn_bbox_loss(inputs[1], inputs[2], outputs[2]))

        return model

@tf.function
def get_proposals(fg_bg_softmaxes, bbox_deltas, anchors, img_sizes, training=None):
    fg_scores = fg_bg_softmaxes[:, :, 1]
    top_n = config.TEST_PRE_NMS_TOP_N if training else config.TEST_PRE_NMS_TOP_N
    _, indices = tf.math.top_k(fg_scores, top_n)

    boxes = []

    for i in range(indices.shape[0]):
        indices_i = indices[i]
        bbox_deltas_i = tf.gather(bbox_deltas[i], indices_i)
        anchors_i = tf.gather(anchors, indices_i)
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
        condition_w = tf.math.logical_and(tf.greater_equal(boxes_i[:, 0], 0), tf.less(boxes_i[:, 2], width))
        condition_h = tf.math.logical_and(tf.greater_equal(boxes_i[:, 1], 0), tf.less(boxes_i[:, 3], height))
        condition = tf.math.logical_and(condition_w, condition_h)
        correct_boxes_indices = tf.squeeze(tf.where(condition), -1)

        boxes_i = tf.gather(boxes_i, correct_boxes_indices)

        if training:
            num_rois = config.TRAIN_POST_NMS_ROIS
            nms_indices = tf.image.non_max_suppression(boxes_i, fg_bg_softmaxes[i, :, 1], num_rois, config.POSITIVE_ANCHOR_THRESHOLD)
        else:
            num_rois = config.TEST_POST_NMS_ROIS
            nms_indices = tf.image.non_max_suppression(boxes_i, fg_bg_softmaxes[i, :, 1], num_rois, config.POSITIVE_ANCHOR_THRESHOLD)

        boxes_i = tf.gather(boxes_i, nms_indices)
        boxes.append(boxes_i)

    boxes = tf.stack(boxes)
    return boxes

if __name__ == "__main__":
    m = RPN(Resnet34_FPN(), 3)

    import numpy as np
    f1, f2, f3 = m.call([np.random.rand(2, 512, 512, 3)])
    print(f1.shape, f2.shape, f3.shape)