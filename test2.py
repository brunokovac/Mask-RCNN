import rpn
import backbone
import xml_util
import image_util
import anchor_utils
import tensorflow as tf
import numpy as np
import os

model = rpn.RPN(backbone.Resnet34_FPN(), 3)
weights_path = "weights.ckpt"
#tf.keras.backend.clear_session()
model.load_weights(weights_path)

anchors = anchor_utils.get_all_anchors((512, 512), [32, 64, 128, 256, 512], [(1, 1), (1, 2), (2, 1)])

path1 = "dataset/VOC2012/JPEGImages/2007_003101.jpg"
path2 = "dataset/VOC2012/Annotations/2007_003101.xml"
img = image_util.load_image(path1)
bboxes, object_classes = xml_util.get_bboxes(path2)
cbs, c, b = model.call([np.array([img]), np.array([object_classes]), np.array([bboxes])])

cls, dlts = anchor_utils.get_rpn_classes_and_bbox_deltas_for_single_image(anchors, bboxes)
print(np.where(cls == 1))

c = tf.keras.backend.eval(c)
b = tf.keras.backend.eval(b)

best_n = 10
for i in range(len(c)):
    cs = c[i]
    bs = b[i]

    vs = cs[:, 1]
    indices = np.argpartition(vs, -best_n)[-best_n:]

    print(vs[indices], cs[:, 0][indices])

    deltas = bs[indices]
    boxes = anchors[indices]

    height = boxes[:, 3] - boxes[:, 1]
    width = boxes[:, 2] - boxes[:, 0]
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 0] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 0] * width
    height *= np.exp(deltas[:, 3])
    width *= np.exp(deltas[:, 2])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    boxes2 = []
    texts = []
    for i in range(len(x1)):
        boxes2.append([x1[i], y1[i], x2[i], y2[i]])
        #boxes2.append([y1[i], x1[i], y2[i], x2[i]])
        texts.append("t" + str(i))
    print(vs[indices])
    print(boxes)

    image_util.draw_bounding_boxes(path1, boxes2, texts)



