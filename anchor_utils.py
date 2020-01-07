import tensorflow as tf
import numpy as np
import config

def get_all_anchors(image_dimensions, scales, ratios):
    anchors = []

    img_width, img_height = image_dimensions
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

                    anchors.append([i-w/2 + j*stride, j-h/2 + i*stride, i+w/2 + j*stride, j+h/2 + i*stride])

    return np.array(anchors)

def get_overlaps(bboxes1, bboxes2):
    result = np.zeros((len(bboxes1), len(bboxes2)))

    for i in range(len(bboxes1)):
        for j in range(len(bboxes2)):
            x1_1, y1_1, x2_1, y2_1 = bboxes1[i]
            x1_2, y1_2, x2_2, y2_2 = bboxes2[j]

            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            overlap = max(0, (min(x2_1, x2_2) - max(x1_1, x1_2))) * max(0, (min(y2_1, y2_2) - max(y1_1, y1_2)))
            result[i][j] = overlap / (area1 + area2 - overlap)

    return result

def get_rpn_classes_and_bbox_deltas_for_single_image2(anchors, gt_bboxes):
    rpn_classes = np.zeros((len(anchors), 1))
    rpn_bbox_deltas = np.zeros((len(anchors), 4))

    overlaps = get_overlaps(anchors, gt_bboxes)

    max_overlaps_by_anchor = np.argmax(overlaps, axis=1)
    max_bboxes = overlaps[range(overlaps.shape[0]), max_overlaps_by_anchor]
    rpn_classes[max_bboxes < config.NEGATIVE_ANCHOR_THRESHOLD] = -1
    rpn_classes[max_bboxes > config.POSITIVE_ANCHOR_THRESHOLD] = 1

    max_overlaps_by_bbox = np.argmax(overlaps, axis=0)
    rpn_classes[max_overlaps_by_bbox] = 1

    MAX_ANCHORS = config.MAX_ANCHORS

    indices = np.where(rpn_classes == -1)[0]
    extras = len(indices) - (MAX_ANCHORS - len(np.where(rpn_classes == 1)[0]))
    if extras > 0:
        rpn_classes[np.random.choice(indices, extras)[0]] = 0

    indices = np.where(rpn_classes == 1)[0]
    extras = len(indices) - MAX_ANCHORS//2
    if extras > 0:
        rpn_classes[np.random.choice(indices, extras)[0]] = 0

    rows = np.where(rpn_classes == 1)[0]
    columns = max_overlaps_by_anchor[rows]
    for i, j in zip(rows, columns):
        anchor = anchors[i]
        gt_bbox = gt_bboxes[j]

        anchor_height = anchor[3] - anchor[1]
        anchor_width = anchor[2] - anchor[0]

        gt_height = gt_bbox[3] - gt_bbox[1]
        gt_width = gt_bbox[2] - gt_bbox[0]

        delta_x = ((gt_bbox[2] - gt_bbox[0])/2 - (anchor[2] - anchor[0])/2) / anchor_width
        delta_y = ((gt_bbox[3] - gt_bbox[1])/2 - (anchor[3] - anchor[1])/2) / anchor_height
        delta_width = np.log(gt_width / anchor_width)
        delta_height = np.log(gt_height / anchor_height)
        rpn_bbox_deltas[i] = [delta_x, delta_y, delta_width, delta_height]

    return rpn_classes, rpn_bbox_deltas

def get_rpn_classes_and_bbox_deltas_for_single_image(anchors, gt_bboxes):
    rpn_classes = np.zeros((len(anchors), 1))
    rpn_bbox_deltas = np.zeros((len(anchors), 4))

    overlaps = get_overlaps(anchors, gt_bboxes)

    max_overlaps_by_anchor = np.argmax(overlaps, axis=1)
    max_bboxes = overlaps[range(overlaps.shape[0]), max_overlaps_by_anchor]
    rpn_classes[max_bboxes < config.NEGATIVE_ANCHOR_THRESHOLD] = -1
    rpn_classes[max_bboxes > config.POSITIVE_ANCHOR_THRESHOLD] = 1

    max_overlaps_by_bbox = np.argmax(overlaps, axis=0)
    rpn_classes[max_overlaps_by_bbox] = 1

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
    for i, j in zip(rows, columns):
        anchor = anchors[i]
        gt_bbox = gt_bboxes[j]

        anchor_height = anchor[3] - anchor[1]
        anchor_width = anchor[2] - anchor[0]

        gt_height = gt_bbox[3] - gt_bbox[1]
        gt_width = gt_bbox[2] - gt_bbox[0]

        delta_x = ((gt_bbox[2] + gt_bbox[0])/2 - (anchor[2] + anchor[0])/2) / anchor_width
        delta_y = ((gt_bbox[3] + gt_bbox[1])/2 - (anchor[3] + anchor[1])/2) / anchor_height
        delta_width = np.log(gt_width / anchor_width)
        delta_height = np.log(gt_height / anchor_height)
        rpn_bbox_deltas[i] = [delta_x, delta_y, delta_width, delta_height]

    return rpn_classes, rpn_bbox_deltas


def get_rpn_classes_and_bbox_deltas(batch_size, anchors, gt_bboxes):
    rpn_classes = np.zeros((batch_size, len(anchors), 1))
    rpn_bbox_deltas = np.zeros((batch_size, len(anchors), 4))

    for i in range(len(gt_bboxes)):
        classes, bboxes = get_rpn_classes_and_bbox_deltas_for_single_image(anchors, gt_bboxes[i])
        rpn_classes[i] = classes
        rpn_bbox_deltas[i] = bboxes

    return rpn_classes, rpn_bbox_deltas

if __name__ == "__main__":
    anchors = get_all_anchors((512, 512), [32, 64, 128, 256, 512], [(1, 1), (1, 2), (2, 1)])
    print(anchors.shape)

    c, b = get_rpn_classes_and_bbox_deltas_for_single_image(anchors, np.array([(174, 101, 349, 351)]))
    print(len(np.where(c == 1)))