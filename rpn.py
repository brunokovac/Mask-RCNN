from backbone import *
import config

class RPN(tf.keras.models.Model):

    def __init__(self, backbone, num_anchors):
        super().__init__()

        self.backbone = backbone
        self.num_anchors = num_anchors

        self.conv = tf.keras.layers.Conv2D(256, (3, 3), padding="same", name="rpn-conv1")
        self.relu = tf.keras.layers.Activation("relu")

        self.fg_bg_conv = tf.keras.layers.Conv2D(self.num_anchors * 2, (1, 1), name="rpn-fg-bg-conv")

        self.fg_bg_softmax = tf.keras.layers.Activation("softmax", name="rpn-fg-bg-softmax")

        self.bbox_conv = tf.keras.layers.Conv2D(self.num_anchors * 4, (1, 1), name="rpn-bbox-conv")

        #return self.compile(tf.keras.optimizers.SGD(1))

    def call(self, inputs, training):
        x = inputs[0]

        fg_bg_softmaxes = []
        fg_bgs = []
        bbox_deltas = []

        levels = self.backbone(x, training)
        for level in levels:
            y = self.conv(level)
            y = self.relu(y)

            fg_bg_conv = self.fg_bg_conv(y)
            fg_bg = tf.reshape(fg_bg_conv, [tf.shape(x)[0], -1, 2])
            fg_bgs.append(fg_bg)

            fg_bg_softmax = self.fg_bg_softmax(fg_bg)
            fg_bg_softmaxes.append(fg_bg_softmax)

            bbox_conv = self.bbox_conv(y)
            bbox = tf.reshape(bbox_conv, [tf.shape(x)[0], -1, 4])
            bbox_deltas.append(bbox)

        fg_bgs = tf.concat(fg_bgs, axis=1)
        fg_bg_softmaxes = tf.concat(fg_bg_softmaxes, axis=1)
        bbox_deltas = tf.concat(bbox_deltas, axis=1)

        return fg_bgs, fg_bg_softmaxes, bbox_deltas, levels

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x, True))

@tf.function
def get_proposals(fg_bg_softmaxes, bbox_deltas, anchors, img_sizes, training=False):
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