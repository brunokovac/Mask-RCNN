import rpn
import backbone
import xml_util
import image_util
import anchor_utils
import numpy as np
import tensorflow as tf
import config
import sys
from PIL import Image, ImageDraw

if len(sys.argv) != 2:
    print("Image name must be provided!")
    sys.exit(0)

image_name = sys.argv[1]

model = rpn.RPN(backbone.Resnet34_FPN(), 3)
weights_path = "weights_all.ckpt"
#tf.keras.backend.clear_session()
model.load_weights(weights_path)

anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

path1 = "DATASET/VOC2012/VOC2012/JPEGImages/" + image_name + ".jpg"
path2 = "DATASET/VOC2012/VOC2012/Annotations/" + image_name + ".xml"
img1, h, w = image_util.load_image(path1)
bboxes, object_classes = xml_util.get_bboxes(path2)
cbs, c, b = model.call([np.array([img1]), np.array([object_classes]), np.array([bboxes])])

cls, dlts = anchor_utils.get_rpn_classes_and_bbox_deltas_for_single_image(anchors, bboxes)

c = tf.keras.backend.eval(c)
b = tf.keras.backend.eval(b)

best_n = 100
for i in range(len(c)):
    cs = c[i]
    bs = b[i]

    #drugo
    indices = np.where(cls == 1)[0]
    deltas = dlts[indices]
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
    #print(cls[indices])

    img = Image.open(path1).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(boxes2)):
        x1, y1, x2, y2 = boxes2[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        #draw.text((x1, y1), texts[i])

    display(img)

    ind = np.where(cls == 1)[0]
    print(len(ind))
    for i in ind:
        print(bs[i], dlts[i], cs[i])

    #prvo
    indices = ind

    deltas = bs[indices]
    boxes = anchors[indices]

    img = Image.open(path1).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        #draw.text((x1, y1), texts[i])

    display(img)

    #drugo2
    indices = ind

    deltas = bs[indices]
    boxes = anchors[indices]
    print(cs[indices])

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

    img = Image.open(path1).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(boxes2)):
        x1, y1, x2, y2 = boxes2[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        #draw.text((x1, y1), texts[i])

    display(img)

    #trece
    vs = cs[:, 1]
    indices = np.argpartition(vs, -100)[-100:]

    deltas = bs[indices]
    boxes = anchors[indices]
    print(cs[indices])

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

    img = Image.open(path1).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(boxes2)):
        x1, y1, x2, y2 = boxes2[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        #draw.text((x1, y1), texts[i])

    display(img)

    # cetvrto

    indices = tf.image.non_max_suppression(boxes2, cs[indices, 1], 10, 0.5).numpy()
    boxes2 = np.array(boxes2)[indices]

    img = Image.open(path1).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(boxes2)):
        x1, y1, x2, y2 = boxes2[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        #draw.text((x1, y1), texts[i])

    display(img)



