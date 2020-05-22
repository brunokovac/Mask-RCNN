import numpy as np
import config

def get_all_anchors(image_dimensions, scales, ratios):
    anchors = []

    img_height, img_width = image_dimensions
    img_width //= 2
    img_height //= 2

    stride = 2

    for scale in scales:
        img_width //= 2
        img_height //= 2
        stride *= 2

        scale **= 2
        for i in range(img_height):
            for j in range(img_width):
                for ratio in ratios:
                    wr, hr = ratio

                    if wr == 1:
                        w = int(round(np.sqrt(scale / hr)))
                        h = int(round(np.sqrt(scale * hr)))
                    elif hr == 1:
                        w = int(round(np.sqrt(scale * wr)))
                        h = int(round(np.sqrt(scale / wr)))

                    x1 = min(max(j * stride - w / 2, 0), image_dimensions[1])
                    y1 = min(max(i * stride - h / 2, 0), image_dimensions[0])
                    x2 = min(max(j * stride + w / 2, 0), image_dimensions[1])
                    y2 = min(max(i * stride + h / 2, 0), image_dimensions[0])

                    '''x1 = j * stride - w / 2
                    y1 = i * stride - h / 2
                    x2 = j * stride + w / 2
                    y2 = i * stride + h / 2'''

                    anchors.append([x1, y1, x2, y2])
                    #anchors.append([j*stride - w/2, i*stride - h/2, j*stride + w/2, i*stride + h/2])

    return np.array(anchors)

def get_overlaps(bboxes1, bboxes2):
    boxes1 = np.repeat(bboxes1, len(bboxes2), axis=0)
    boxes2 = np.tile(bboxes2, (len(bboxes1), 1))

    x1_1, x2_1 = boxes1[:, 0], boxes2[:, 0]
    x1_2, x2_2 = boxes1[:, 2], boxes2[:, 2]
    y1_1, y2_1 = boxes1[:, 1], boxes2[:, 1]
    y1_2, y2_2 = boxes1[:, 3], boxes2[:, 3]

    w = np.maximum(0, np.minimum(x1_2, x2_2) - np.maximum(x1_1, x2_1))
    h = np.maximum(0, np.minimum(y1_2, y2_2) - np.maximum(y1_1, y2_1))
    intersection = w * h

    area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
    area2 = (x2_2 - x2_1) * (y2_2 - y2_1)

    iou = intersection / (area1 + area2 - intersection)
    iou = iou.reshape((len(bboxes1), len(bboxes2)))

    return iou

def get_deltas(anchors, gt_bboxes):
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_widths = anchors[:, 2] - anchors[:, 0]

    gt_heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    gt_widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]

    delta_x = ((gt_bboxes[:, 2] + gt_bboxes[:, 0]) / 2 - (anchors[:, 2] + anchors[:, 0]) / 2) / anchor_widths
    delta_y = ((gt_bboxes[:, 3] + gt_bboxes[:, 1]) / 2 - (anchors[:, 3] + anchors[:, 1]) / 2) / anchor_heights
    delta_width = np.log(gt_widths / anchor_widths)
    delta_height = np.log(gt_heights / anchor_heights)

    res = np.stack([delta_x, delta_y, delta_width, delta_height], axis=1)
    return res

def get_rpn_classes_and_bbox_deltas_for_single_image(anchors, gt_bboxes):
    rpn_classes = np.zeros(len(anchors))
    rpn_bbox_deltas = np.zeros((len(anchors), 4))

    gt_bboxes = gt_bboxes[np.sum(gt_bboxes, axis=1) > 0]
    overlaps = get_overlaps(anchors, gt_bboxes)

    max_overlaps_by_anchor = np.argmax(overlaps, axis=1)
    max_bboxes = overlaps[range(overlaps.shape[0]), max_overlaps_by_anchor]
    rpn_classes[max_bboxes < config.NEGATIVE_ANCHOR_THRESHOLD] = -1
    rpn_classes[max_bboxes > config.POSITIVE_ANCHOR_THRESHOLD] = 1

    max_overlaps_by_bbox = np.max(overlaps, axis=0)
    rpn_classes[np.argwhere(overlaps == max_overlaps_by_bbox)[:, 0]] = 1

    MAX_ANCHORS = config.MAX_ANCHORS

    indices = np.where(rpn_classes == 1)[0]
    extras = len(indices) - MAX_ANCHORS // 2
    if extras > 0:
        rpn_classes[np.random.choice(indices, extras, replace=False)] = 0

    indices = np.where(rpn_classes == -1)[0]
    extras = len(indices) - (MAX_ANCHORS - len(np.where(rpn_classes == 1)[0]))
    if extras > 0:
        rpn_classes[np.random.choice(indices, extras, replace=False)] = 0

    rows = np.where(rpn_classes == 1)[0]
    columns = max_overlaps_by_anchor[rows]

    selected_anchors = anchors[rows]
    selected_gt_bboxes = gt_bboxes[columns]
    rpn_bbox_deltas[rows] = get_deltas(selected_anchors, selected_gt_bboxes)

    return rpn_classes, rpn_bbox_deltas

def get_rpn_classes_and_bbox_deltas(batch_size, anchors, gt_bboxes):
    rpn_classes = np.zeros((batch_size, len(anchors)))
    rpn_bbox_deltas = np.zeros((batch_size, len(anchors), 4))

    for i in range(len(gt_bboxes)):
        classes, bboxes = get_rpn_classes_and_bbox_deltas_for_single_image(anchors, gt_bboxes[i])
        rpn_classes[i] = classes
        rpn_bbox_deltas[i] = bboxes

    return rpn_classes, rpn_bbox_deltas

if __name__ == "__main__":
    import anchor_utils
    import dataset_util
    import image_util

    #np.random.seed(104)
    anchors = anchor_utils.get_all_anchors((512, 512), [32, 64, 128, 256, 512], [(1, 1), (1, 2), (2, 1)])

    # ds = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 2)
    ds = dataset_util.Dataset("dataset/VOC2012", "/train_list.txt", 1)
    data1, data2_2, data3, d4 = ds.next_batch()
    print(data2_2)
    print(data3)
    data2_2, data3_2 = anchor_utils.get_rpn_classes_and_bbox_deltas(len(data1), anchors, data2_2)

    ind = np.where(data2_2[0] == 1)[0]
    boxes = anchors[ind]
    print(len(ind))

    #image_util.draw_bounding_boxes_from_array(data1[0].astype(np.uint8), data2_2[0], ["t" for i in range(len(data3[0]))])
    #image_util.draw_bounding_boxes_from_array(data1[0].astype(np.uint8), boxes, ["t" for i in range(len(boxes))])
    #n = 17500
    #boxes = anchors[n*3:n*3+3]
    image_util.draw_bounding_boxes_from_array(data1[0].astype(np.uint8), boxes)
