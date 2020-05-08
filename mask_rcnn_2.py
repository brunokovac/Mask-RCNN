import tensorflow as tf
import numpy as np
from backbone import *
import rpn
import anchor_utils
import xml_util
import dataset_util
import config

def mask_rcnn_class_loss(gt_classes, predicted_classes_softmax):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_classes, predicted_classes_softmax)
    return tf.reduce_mean(loss)

def rpn_bbox_loss(gt_classes, gt_bboxes, predicted_bboxes):
    indices = tf.where(tf.greater(gt_classes, 0))

    gt_bboxes_selected = tf.gather_nd(gt_bboxes, indices)
    pred_bboxes_selected = tf.gather_nd(predicted_bboxes, indices)

    diff = tf.abs(gt_bboxes_selected - pred_bboxes_selected)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_mean(loss)
    return loss

def get_overlaps(bboxes1, bboxes2):
    boxes1 = tf.keras.backend.repeat_elements(bboxes1, len(bboxes2), axis=0)
    boxes2 = tf.tile(bboxes2, (len(bboxes1), 1))

    x1_1, x2_1 = boxes1[:, 0], boxes2[:, 0]
    x1_2, x2_2 = boxes1[:, 2], boxes2[:, 2]
    y1_1, y2_1 = boxes1[:, 1], boxes2[:, 1]
    y1_2, y2_2 = boxes1[:, 3], boxes2[:, 3]

    w = tf.math.maximum(0, np.minimum(x1_2, x2_2) - tf.math.maximum(x1_1, x2_1))
    h = tf.math.maximum(0, np.minimum(y1_2, y2_2) - tf.math.maximum(y1_1, y2_1))
    intersection = w * h

    area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
    area2 = (x2_2 - x2_1) * (y2_2 - y2_1)

    iou = intersection / (area1 + area2 - intersection)
    iou = tf.reshape(iou, [len(bboxes1), len(bboxes2)])

    return iou

def get_deltas(proposals, gt_boxes):
    proposals_height = proposals[:, 3] - proposals[:, 1]
    proposals_width = proposals[:, 2] - proposals[:, 0]

    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]

    delta_x = ((gt_boxes[:, 2] + gt_boxes[:, 0]) / 2 - (proposals[:, 2] + proposals[:, 0]) / 2) / proposals_width
    delta_y = ((gt_boxes[:, 3] + gt_boxes[:, 1]) / 2 - (proposals[:, 3] + proposals[:, 1]) / 2) / proposals_height
    delta_width = np.log(gt_widths / proposals_width)
    delta_height = np.log(gt_heights / proposals_height)

    return tf.stack([delta_x, delta_y, delta_width, delta_height], axis=1)

def generate_mask_rcnn_labels(proposals, gt_classes, gt_bboxes):
    mask_rcnn_classes = []
    mask_rcnn_deltas = []
    mask_rcnn_masks = []

    for i in range(len(proposals)):
        proposals_i = proposals[i]
        gt_classes_i = gt_classes[i]
        gt_bboxes_i = gt_bboxes[i]

        overlaps = get_overlaps(proposals_i, gt_bboxes_i)

        max_overlap_by_proposal_indices = tf.math.argmax(overlaps, axis=1)
        max_overlap_by_proposal = tf.math.reduce_max(overlaps, axis=1)

        proposal_i_classes = tf.where(max_overlap_by_proposal > 0.5,
                                      tf.gather(gt_classes_i, max_overlap_by_proposal_indices),
                                      tf.zeros(len(proposals_i), dtype="int32"))
        mask_rcnn_classes.append(proposal_i_classes)

        matched_boxes = tf.gather(gt_bboxes_i, max_overlap_by_proposal_indices)
        mask_rcnn_deltas.append(get_deltas(proposals_i, matched_boxes))


    return tf.convert_to_tensor(mask_rcnn_classes), tf.convert_to_tensor(mask_rcnn_deltas), tf.convert_to_tensor(mask_rcnn_masks)

class Mask_RCNN(tf.keras.models.Model):

    def __init__(self, rpn, anchors, num_classes):
        super().__init__()

        self.rpn = rpn
        self.anchors = anchors

        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(1024)
        self.relu1 = tf.keras.layers.Activation("relu")

        self.fc2 = tf.keras.layers.Dense(1024)
        self.relu2 = tf.keras.layers.Activation("relu")

        self.classes_logits = tf.keras.layers.Dense(num_classes)
        self.classes_softmax = tf.keras.layers.Activation("softmax")

        self.bboxes = tf.keras.layers.Dense(num_classes * 4)

        return self.compile(tf.keras.optimizers.SGD(1))

    def call(self, inputs, training=False):
        return self.train(inputs) if training else self.eval(inputs)

    def train(self, inputs):
        x = inputs[0]
        labels = inputs[1]
        img_sizes = inputs[2]

        fg_bgs, fg_bg_softmaxes, bbox_deltas, levels = self.rpn(inputs, True)
        proposals = self.get_proposals_for_batch(fg_bg_softmaxes, bbox_deltas, img_sizes, True)
        rois = self.map_proposals_to_fpn_levels(proposals, levels)

        y = self.flatten(rois)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)

        classes_logits = self.classes_logits(y)
        classes_softmax = self.classes_softmax(classes_logits)

        bboxes = self.bboxes(y)

        #mask_rcnn_classes, mask_rcnn_deltas, mask_rcnn_masks = generate_mask_rcnn_labels(bboxes, labels[0], la)

        return classes_softmax, bboxes

    def eval(self, inputs):
        x = inputs[0]
        img_sizes = inputs[1]

        fg_bgs, fg_bg_softmaxes, bbox_deltas, levels = self.rpn(inputs, False)
        proposals = self.get_proposals_for_batch(fg_bg_softmaxes, bbox_deltas, img_sizes, False)
        rois = self.map_proposals_to_fpn_levels(proposals, levels)

        y = self.flatten(rois)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)

        classes_logits = self.classes_logits(y)
        classes_softmax = self.classes_softmax(classes_logits)

        bboxes = self.bboxes(y)

        return classes_softmax, bboxes

    def get_proposals(self, inputs, training):
        if training:
            return self.get_proposals_for_batch(inputs, training)

        proposals = []
        x = inputs[0]
        for i in range(x.shape[0]):
            x_i = x[i]
            proposals_i = self.get_proposals_for_batch([x_i, inputs[1][i]], training)
            proposals.append(proposals_i[0])

        return proposals

    def get_proposals_for_batch(self, fg_bg_softmaxes, bbox_deltas, img_sizes, training):
        fg_scores = fg_bg_softmaxes[:, :, 1]
        top_n = config.TEST_PRE_NMS_TOP_N_PER_IMAGE if training else config.TEST_PRE_NMS_TOP_N_PER_IMAGE
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
            correct_boxes_indices = tf.where(tf.greater((boxes_i[:, 2] - boxes_i[:, 0]) * (boxes_i[:, 3] - boxes_i[:, 1]), 0))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)

            nms_scores = tf.gather_nd(fg_scores_i, correct_boxes_indices)

            if training:
                num_rois = config.TRAIN_POST_NMS_TOP_N_PER_IMAGE
                nms_indices = tf.image.non_max_suppression(boxes_i, nms_scores, num_rois, iou_threshold=0.7)
            else:
                num_rois = config.TEST_POST_NMS_TOP_N_PER_IMAGE
                nms_indices = tf.image.non_max_suppression(boxes_i, nms_scores, num_rois, iou_threshold=0.7)

            boxes_i = tf.gather(boxes_i, nms_indices)
            boxes_i = tf.pad(boxes_i, [[0, tf.math.maximum(num_rois - tf.shape(boxes_i)[0], 0)], [0, 0]], constant_values=0)
            boxes.append(boxes_i)

        return tf.convert_to_tensor(boxes)

    # ROI align from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
    def map_proposals_to_fpn_levels(self, proposals, levels):
        boxes = tf.stack([proposals[:, :, 1], proposals[:, :, 0], proposals[:, :, 3], proposals[:, :, 2]], axis=2)

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = levels

        logs = tf.math.log((proposals[:, :, 2] - proposals[:, :, 0])
                           * (proposals[:, :, 3] - proposals[:, :, 1]) / 224) / tf.math.log(2.0)
        mappings = tf.cast(tf.math.minimum(tf.math.floor(tf.math.maximum(logs, 2)), 5), "int32")
        roi_level = mappings

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, (7, 7),
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def map_proposals_to_fpn_levels_2(self, proposals, levels):
        rois_by_images = [[] for _ in range(proposals.shape[0])]

        logs = tf.math.log((proposals[:, :, 2] - proposals[:, :, 0])
                           * (proposals[:, :, 3] - proposals[:, :, 1]) / 224) / tf.math.log(2.0)
        mappings = tf.cast(tf.math.minimum(tf.math.floor(tf.math.maximum(logs, 2)), 5), "int32")
        #scaled_proposals = proposals / tf.expand_dims(2 ** mappings, 2)

        rearranged_proposals = tf.stack([proposals[:, :, 1], proposals[:, :, 0], proposals[:, :, 3], proposals[:, :, 2]], axis=2)
        for i, level in enumerate(levels):
            indices = tf.cast(tf.where(tf.equal(mappings, i + 2)), "int32")
            level_boxes = tf.gather_nd(rearranged_proposals, indices) / config.IMAGE_SIZE[0]
            new_rois = tf.image.crop_and_resize(levels[i], level_boxes,
                                                indices[:, 0],
                                                [7, 7], method="bilinear")
            print(new_rois.shape)
            start = 0
            for j in range(proposals.shape[0]):
                count = tf.where(tf.equal(indices[:, 0], j)).shape[0]
                rois_by_images[j].extend(new_rois[start : start + count])
                start = count

        print("kraj")
        rois = tf.ragged.constant(rois_by_images)
        print(rois.shape)
        return

if __name__ == "__main__":
    np.random.seed(100)
    tf.random.set_seed(110)
    anchors = anchor_utils.get_all_anchors((512, 512), [64, 128, 256, 512, 1024], [(1, 1), (1, 2), (2, 1)])
    rpn_model = rpn.RPN(Resnet34_FPN(), 3)
    #weights_path = "weights.ckpt"
    #rpn_model.load_weights(weights_path)
    num_classes = len(xml_util.classes)
    model = Mask_RCNN(rpn_model, anchors, num_classes)

    #ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 2)
    ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 2)
    data1, data2, data3, d4 = ds.next_batch()
    data2, data3 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, data2)

    classes, bboxes = model.call([data1, d4], training=True)