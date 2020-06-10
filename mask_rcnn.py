from backbone import *
import rpn
import anchor_utils
import dataset_util
import config

def pad_softmaxes_with_zero_class(tensor, num_to_pad):
    ones_padding = tf.ones([num_to_pad, 1])
    zeros_padding = tf.zeros([num_to_pad, len(config.CLASSES) - 1])
    padding = tf.concat([ones_padding, zeros_padding], axis=1)
    return tf.concat([tensor, padding], axis=0)

def pad_boxes_per_class_with_zeros(tensor, num_to_pad):
    return tf.pad(tensor, [[0, num_to_pad], [0, 0], [0, 0]], constant_values=0)

def pad_boxes_with_zeros(tensor, num_to_pad):
    return tf.pad(tensor, [[0, num_to_pad], [0, 0]], constant_values=0)

def pad_classes_with_zeros(tensor, num_to_pad):
    return tf.pad(tensor, [[0, num_to_pad]], constant_values=0)

def pad_masks_with_zeros(tensor, num_to_pad):
    return tf.pad(tensor, [[0, num_to_pad], [0, 0], [0, 0]], constant_values=0)

def pad_masks_per_class_with_zeros(tensor, num_to_pad):
    return tf.pad(tensor, [[0, num_to_pad], [0, 0], [0, 0], [0, 0]], constant_values=0)

def get_overlaps(bboxes1, bboxes2):
    boxes1 = tf.reshape(tf.cast(tf.tile(bboxes1, (1, tf.shape(bboxes2)[0])), "float32"), [-1, 4])
    boxes2 = tf.cast(tf.tile(bboxes2, (tf.shape(bboxes1)[0], 1)), "float32")

    x1_1, x2_1 = boxes1[:, 0], boxes2[:, 0]
    x1_2, x2_2 = boxes1[:, 2], boxes2[:, 2]
    y1_1, y2_1 = boxes1[:, 1], boxes2[:, 1]
    y1_2, y2_2 = boxes1[:, 3], boxes2[:, 3]

    w = tf.math.maximum(0.0, tf.math.minimum(x1_2, x2_2) - tf.math.maximum(x1_1, x2_1))
    h = tf.math.maximum(0.0, tf.math.minimum(y1_2, y2_2) - tf.math.maximum(y1_1, y2_1))
    intersection = w * h

    area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
    area2 = (x2_2 - x2_1) * (y2_2 - y2_1)

    iou = intersection / (area1 + area2 - intersection)
    iou = tf.reshape(iou, [tf.shape(bboxes1)[0], tf.shape(bboxes2)[0]])

    return iou

def get_deltas(proposals, gt_boxes):
    gt_boxes = tf.cast(gt_boxes, "float32")

    proposals_height = proposals[:, 3] - proposals[:, 1]
    proposals_width = proposals[:, 2] - proposals[:, 0]

    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]

    delta_x = ((gt_boxes[:, 2] + gt_boxes[:, 0]) / 2 - (proposals[:, 2] + proposals[:, 0]) / 2) / proposals_width
    delta_y = ((gt_boxes[:, 3] + gt_boxes[:, 1]) / 2 - (proposals[:, 3] + proposals[:, 1]) / 2) / proposals_height
    delta_width = tf.math.log(gt_widths / proposals_width)
    delta_height = tf.math.log(gt_heights / proposals_height)

    return tf.stack([delta_x, delta_y, delta_width, delta_height], axis=1)

def generate_mask_rcnn_labels(proposals, predicted_classes, predicted_bbox_deltas, predicted_masks, gt_classes, gt_bboxes, gt_masks):
    mask_rcnn_proposals = []
    mask_rcnn_predicted_classes = []
    mask_rcnn_predicted_bbox_deltas = []
    mask_rcnn_predicted_masks = []
    mask_rcnn_classes = []
    mask_rcnn_deltas = []
    mask_rcnn_masks = []

    for i in range(proposals.shape[0]):
        non_padded_indices = tf.where(tf.reduce_sum(tf.abs(proposals[i]), axis=1) > 0)
        proposals_i = tf.gather_nd(proposals[i], non_padded_indices)

        predicted_classes_i = predicted_classes[i]
        predicted_bbox_deltas_i = predicted_bbox_deltas[i]
        predicted_masks_i = predicted_masks[i]

        non_padded_indices = tf.where(tf.reduce_sum(tf.abs(gt_bboxes[i]), axis=1) > 0)
        gt_classes_i = tf.gather_nd(gt_classes[i], non_padded_indices)
        gt_bboxes_i = tf.gather_nd(gt_bboxes[i], non_padded_indices)
        gt_masks_i = tf.gather_nd(gt_masks[i], non_padded_indices)

        overlaps = get_overlaps(proposals_i, gt_bboxes_i)

        max_overlap_by_proposal_indices = tf.math.argmax(overlaps, axis=1)
        max_overlap_by_proposal = tf.math.reduce_max(overlaps, axis=1)

        positive_roi_indices = tf.where(max_overlap_by_proposal > config.POSITIVE_ROI_THRESHOLD)
        num_positives = int(config.POSITIVE_ROIS_RATIO * config.MAX_ROIS)
        positive_roi_indices = tf.random.shuffle(positive_roi_indices)[:num_positives]

        negative_roi_indices = tf.where(max_overlap_by_proposal < config.POSITIVE_ROI_THRESHOLD)
        num_negatives = tf.cast((int(1 / config.POSITIVE_ROIS_RATIO) - 1) * tf.shape(positive_roi_indices)[0], "int32")
        negative_roi_indices = tf.random.shuffle(negative_roi_indices)[:num_negatives]

        all_indices = tf.concat([positive_roi_indices, negative_roi_indices], axis=0)
        selected_proposals_i = tf.gather_nd(proposals_i, all_indices)
        mask_rcnn_proposals.append(pad_boxes_with_zeros(selected_proposals_i, config.MAX_ROIS - tf.shape(all_indices)[0]))

        gt_classes_indices = tf.gather_nd(max_overlap_by_proposal_indices, positive_roi_indices)
        proposal_i_classes = tf.gather(gt_classes_i, gt_classes_indices)
        mask_rcnn_classes.append(pad_classes_with_zeros(proposal_i_classes, config.MAX_ROIS - tf.shape(positive_roi_indices)[0]))

        proposal_i_predicted_classes = tf.gather_nd(predicted_classes_i, all_indices)
        mask_rcnn_predicted_classes.append(pad_softmaxes_with_zero_class(proposal_i_predicted_classes, config.MAX_ROIS - tf.shape(all_indices)[0]))

        matched_gt_boxes = tf.gather(gt_bboxes_i, gt_classes_indices)
        positive_proposals_i = tf.gather_nd(proposals_i, positive_roi_indices)
        gt_bbox_deltas = get_deltas(positive_proposals_i, matched_gt_boxes)
        mask_rcnn_deltas.append(pad_boxes_with_zeros(gt_bbox_deltas, config.MAX_ROIS - tf.shape(positive_roi_indices)[0]))

        selected_predicted_bbox_deltas = tf.gather_nd(predicted_bbox_deltas_i, positive_roi_indices)
        mask_rcnn_predicted_bbox_deltas.append(pad_boxes_per_class_with_zeros(selected_predicted_bbox_deltas, config.MAX_ROIS - tf.shape(positive_roi_indices)[0]))

        matched_gt_masks = tf.gather(gt_masks_i, gt_classes_indices)
        mask_rcnn_masks.append(pad_masks_with_zeros(matched_gt_masks, config.MAX_ROIS - tf.shape(positive_roi_indices)[0]))

        selected_predicted_masks = tf.gather_nd(predicted_masks_i, positive_roi_indices)
        mask_rcnn_predicted_masks.append(pad_masks_per_class_with_zeros(selected_predicted_masks, config.MAX_ROIS - tf.shape(positive_roi_indices)[0]))

    return tf.convert_to_tensor(mask_rcnn_proposals), tf.convert_to_tensor(mask_rcnn_predicted_classes), \
           tf.convert_to_tensor(mask_rcnn_predicted_bbox_deltas), tf.convert_to_tensor(mask_rcnn_predicted_masks), \
           tf.convert_to_tensor(mask_rcnn_classes), tf.convert_to_tensor(mask_rcnn_deltas), tf.convert_to_tensor(mask_rcnn_masks)

class Mask_RCNN(tf.keras.models.Model):

    def __init__(self, rpn, anchors, num_classes):
        super().__init__()

        self.rpn = rpn
        self.anchors = anchors
        self.num_classes = num_classes

        self.fc1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, name="mask-fc1"), name="mask-td1")
        self.bn1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(name="mask-bn1"), name="mask-td2")
        self.relu1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"))

        self.fc2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, name="mask-fc2"), name="mask-td3")
        self.bn2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(name="mask-bn2"), name="mask-td4")
        self.relu2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"))

        self.classes_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, name="mask-classes-logits"), name="mask-td5")
        self.classes_softmax = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax"))

        self.bbox_deltas = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes * 4, name="mask-bbox-deltas"), name="mask-td6")

        # mask head
        self.mask_head = []
        for i in range(4):
            self.mask_head.append(tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(256, (3, 3), padding="same", name="mask-conv{}".format(i)), name="mask-head-conv-td{}".format(i)))
            self.mask_head.append(tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization(name="mask-head-bn{}".format(i)), name="mask-head-bn-td{}".format(i)))
            self.mask_head.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu")))

        self.mask_deconv = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation="relu", name="mask-deconv1"), name="mask-head-td5")
        self.mask_conv5 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(num_classes, (1, 1), activation="sigmoid", name="mask-conv5"), name="mask-head-td6")

        #return self.compile(tf.keras.optimizers.SGD(1))

    def call(self, inputs, training):
        return self.train(inputs) if training else self.eval(inputs)

    def forward_pass(self, inputs, training):
        x = inputs[0]
        img_sizes = inputs[1]

        fg_bgs, fg_bg_softmaxes, rpn_bbox_deltas, levels = self.rpn(inputs, training)

        num_rois = config.TRAIN_POST_NMS_TOP_N_PER_IMAGE if training else config.TEST_POST_NMS_TOP_N_PER_IMAGE
        proposals = self.get_proposals_for_batch(fg_bg_softmaxes, rpn_bbox_deltas, img_sizes, training)
        proposals = tf.reshape(proposals, [x.shape[0], num_rois, 4])
        rois_shape = config.ROIS_SHAPE
        rois = self.map_proposals_to_fpn_levels(proposals, levels, rois_shape)

        y = tf.reshape(rois, [x.shape[0], num_rois, config.FPN_NUM_CHANNELS * rois_shape[0] * rois_shape[1]])
        y = self.fc1(y)
        y = self.bn1(y, training=training)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.bn2(y, training=training)
        y = self.relu2(y)

        classes_logits = self.classes_logits(y)
        classes_softmax = self.classes_softmax(classes_logits)

        bbox_deltas = self.bbox_deltas(y)
        bbox_deltas = tf.reshape(bbox_deltas, [x.shape[0], num_rois, self.num_classes, 4])

        # mask head
        mask_rois_shape = config.MASK_ROIS_SHAPE
        top_n_mask_proposals = config.TOP_N_MASK_PROPOSALS if training else config.TEST_POST_NMS_TOP_N_PER_IMAGE
        mask_rois = self.map_proposals_to_fpn_levels(proposals[:, :top_n_mask_proposals], levels, mask_rois_shape)

        y = mask_rois
        for layer in self.mask_head:
            y = layer(y)
        y = self.mask_deconv(y)
        masks = self.mask_conv5(y)
        masks = tf.transpose(masks, [0, 1, 4, 2, 3])

        return fg_bg_softmaxes, rpn_bbox_deltas, classes_softmax, bbox_deltas, masks, proposals

    def train(self, inputs):
        return self.forward_pass(inputs, True)

    def eval(self, inputs):
        images = inputs[0]
        img_sizes = inputs[1]
        fg_bg_softmaxes, rpn_bbox_deltas, classes_softmax, bbox_deltas, pred_masks, proposals = self.forward_pass(inputs, False)

        bboxes = []
        masks = []
        classes_scores = []
        classes = []
        for i in range(tf.shape(images)[0]):
            boxes_i = []
            classes_scores_i = []
            classes_i = []
            masks_i = []
            for j in range(1, len(config.CLASSES)):
                indices = tf.where(classes_softmax[i, :, j] > config.TEST_CLASSIFICATION_SCORE_THRESHOLD)
                if indices.shape[0] == 0:
                    continue

                classes_softmax_j = tf.gather_nd(classes_softmax[i, :, j], indices)

                proposals_j = tf.gather_nd(proposals[i], indices)
                bbox_deltas_j = tf.gather_nd(bbox_deltas[i, :, j], indices)
                masks_j = tf.gather_nd(pred_masks[i, :, j], indices)

                bboxes_i_j, masks_i_j, classes_softmax_i_j = self.get_bboxes_for_class(proposals_j, bbox_deltas_j, masks_j, classes_softmax_j, img_sizes[i])
                classes_scores_i.extend(classes_softmax_i_j)
                classes_i.extend(tf.ones(bboxes_i_j.shape[0], dtype="int32") * j)
                boxes_i.extend(bboxes_i_j)

                for k in range(masks_i_j.shape[0]):
                    x1, y1, x2, y2 = bboxes_i_j[k]
                    h, w = y2 - y1 + 1, x2 - x1 + 1
                    masks_i.append(tf.cast(tf.image.resize(tf.expand_dims(masks_i_j[k], axis=2), (h, w))[:-1, :-1] > config.MASK_THRESHOLD, "int32"))

            bboxes.append(boxes_i)
            classes_scores.append(classes_scores_i)
            classes.append(classes_i)
            masks.append(masks_i)

        return bboxes, classes_scores, classes, masks, fg_bg_softmaxes, rpn_bbox_deltas, classes_softmax, bbox_deltas, pred_masks, proposals

    def get_bboxes_for_class(self, proposals, bbox_deltas, masks, class_softmax, img_sizes):
        img_sizes = tf.cast(img_sizes, "float32")

        height = proposals[:, 3] - proposals[:, 1]
        width = proposals[:, 2] - proposals[:, 0]
        center_y = proposals[:, 1] + 0.5 * height
        center_x = proposals[:, 0] + 0.5 * width

        center_y += bbox_deltas[:, 1] * height
        center_x += bbox_deltas[:, 0] * width
        height *= tf.exp(bbox_deltas[:, 3])
        width *= tf.exp(bbox_deltas[:, 2])

        y1 = tf.math.minimum(tf.math.maximum(center_y - 0.5 * height, 0), img_sizes[0])
        x1 = tf.math.minimum(tf.math.maximum(center_x - 0.5 * width, 0), img_sizes[1])
        y2 = tf.math.minimum(tf.math.maximum(y1 + height, 0), img_sizes[0])
        x2 = tf.math.minimum(tf.math.maximum(x1 + width, 0), img_sizes[1])

        bboxes = tf.stack([x1, y1, x2, y2], axis=1)

        correct_boxes_indices = tf.where((bboxes[:, 2] - bboxes[:, 0]) > 0)
        bboxes = tf.gather_nd(bboxes, correct_boxes_indices)
        correct_boxes_indices = tf.where((bboxes[:, 3] - bboxes[:, 1]) > 0)
        bboxes = tf.gather_nd(bboxes, correct_boxes_indices)

        nms_scores = tf.gather_nd(class_softmax, correct_boxes_indices)
        nms_indices = tf.image.non_max_suppression(bboxes, nms_scores, config.TEST_MAX_BOXES_PER_CLASS,
                                                   iou_threshold=config.TEST_NMS_IOU_THRESHOLD)

        bboxes = tf.gather(bboxes, nms_indices)
        class_softmax = tf.gather(class_softmax, nms_indices)
        masks = tf.gather(masks, nms_indices)
        return tf.cast(tf.round(bboxes), "int32"), masks, class_softmax

    def get_proposals(self, inputs, training):
        if training:
            return self.get_proposals_for_batch(inputs, training)

        proposals = []
        x = inputs[0]
        for i in range(tf.shape(x)[0]):
            x_i = x[i]
            proposals_i = self.get_proposals_for_batch([x_i, inputs[1][i]], training)
            proposals.append(proposals_i[0])

        return proposals

    def get_proposals_for_batch(self, fg_bg_softmaxes, bbox_deltas, img_sizes, training):
        fg_scores = fg_bg_softmaxes[:, :, 1]
        top_n = config.TRAIN_PRE_NMS_TOP_N_PER_IMAGE if training else config.TEST_PRE_NMS_TOP_N_PER_IMAGE
        _, indices = tf.math.top_k(fg_scores, top_n, sorted=True)

        boxes = []

        for i in range(indices.shape[0]):
            indices_i = indices[i]
            fg_scores_i = fg_scores[i]
            bbox_deltas_i = tf.gather(bbox_deltas[i], indices_i)
            anchors_i = tf.gather(self.anchors, indices_i)
            anchors_i = tf.cast(anchors_i, "float32")
            img_size_i = tf.cast(img_sizes[i], "float32")

            height = anchors_i[:, 3] - anchors_i[:, 1]
            width = anchors_i[:, 2] - anchors_i[:, 0]
            center_y = anchors_i[:, 1] + 0.5 * height
            center_x = anchors_i[:, 0] + 0.5 * width

            center_y += bbox_deltas_i[:, 1] * height
            center_x += bbox_deltas_i[:, 0] * width
            height *= tf.exp(bbox_deltas_i[:, 3])
            width *= tf.exp(bbox_deltas_i[:, 2])

            y1 = tf.math.minimum(tf.math.maximum(center_y - 0.5 * height, 0), img_size_i[0])
            x1 = tf.math.minimum(tf.math.maximum(center_x - 0.5 * width, 0), img_size_i[1])
            y2 = tf.math.minimum(tf.math.maximum(y1 + height, 0), img_size_i[0])
            x2 = tf.math.minimum(tf.math.maximum(x1 + width, 0),  img_size_i[1])

            '''y1 = center_y - 0.5 * height
            x1 = center_x - 0.5 * width
            y2 = y1 + height
            x2 = x1 + width'''

            boxes_i = tf.stack([x1, y1, x2, y2], axis=1)

            correct_boxes_indices = tf.where((boxes_i[:, 2] - boxes_i[:, 0]) > 0)
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where((boxes_i[:, 3] - boxes_i[:, 1]) > 0)
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)

            '''height = tf.cast(img_sizes[i, 0], "float32")
            width = tf.cast(img_sizes[i, 1], "float32")
            correct_boxes_indices = tf.where(tf.greater(boxes_i[:, 0], 0))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.less(boxes_i[:, 2], width))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.greater_equal(boxes_i[:, 1], 0))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.less(boxes_i[:, 3], height))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)
            correct_boxes_indices = tf.where(tf.greater((boxes_i[:, 2] - boxes_i[:, 0]) * (boxes_i[:, 3] - boxes_i[:, 1]), 0))
            boxes_i = tf.gather_nd(boxes_i, correct_boxes_indices)'''

            nms_scores = tf.gather_nd(fg_scores_i, correct_boxes_indices)

            num_rois = config.TRAIN_POST_NMS_TOP_N_PER_IMAGE if training else config.TEST_POST_NMS_TOP_N_PER_IMAGE
            nms_indices = tf.image.non_max_suppression(boxes_i, nms_scores, num_rois, iou_threshold=0.7)

            boxes_i = tf.gather(boxes_i, nms_indices)
            boxes_i = pad_boxes_with_zeros(boxes_i, tf.math.maximum(num_rois - tf.shape(boxes_i)[0], 0))
            #print(boxes_i)
            boxes.append(boxes_i)

        return tf.convert_to_tensor(boxes)

    # ROI align from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
    def map_proposals_to_fpn_levels(self, proposals, levels, rois_shape):
        boxes = tf.stack([proposals[:, :, 1], proposals[:, :, 0], proposals[:, :, 3], proposals[:, :, 2]], axis=2)

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = levels

        logs = tf.math.log(tf.math.sqrt((proposals[:, :, 2] - proposals[:, :, 0])
                           * (proposals[:, :, 3] - proposals[:, :, 1])) / 224) / tf.math.log(2.0)
        mappings = tf.cast(tf.math.minimum(tf.math.round(tf.math.maximum(4 + logs, 2)), 5), "int32")
        roi_level = mappings

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix) / config.IMAGE_SIZE[0]

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

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
            pooled.append(tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, rois_shape, method="bilinear"))

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
    #np.random.seed(100)
    tf.random.set_seed(110)
    anchors = anchor_utils.get_all_anchors((512, 512), [64, 128, 256, 512, 1024], [(1, 1), (1, 2), (2, 1)])
    rpn_model = rpn.RPN(Resnet34_FPN(), 3)
    #weights_path = "weights.ckpt"
    #rpn_model.load_weights(weights_path)
    num_classes = len(config.CLASSES)
    model = Mask_RCNN(rpn_model, anchors, num_classes)

    #ds = dataset_util.VOC2012_Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 2)
    ds = dataset_util.VOC2012_Dataset("dataset/VOC2012", "/train_list.txt", 2)
    data1, data2, data3, data5, d4 = ds.next_batch()
    data2, data3 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, data2)

    a, b, c, d, e = model([data1, d4], training=True)