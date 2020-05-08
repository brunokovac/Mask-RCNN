import xml.etree.ElementTree as ET
import numpy as np
import config

def get_bboxes(path):
    tree = ET.parse(path)
    root = tree.getroot()

    bboxes = np.zeros([config.MAX_OBJECTS_PER_IMAGE, 4], dtype="int32")
    object_classes = np.zeros([config.MAX_OBJECTS_PER_IMAGE], dtype="int32")

    for i, boxes in enumerate(root.iter('object')):

        class_name = boxes.find("name").text

        for box in boxes.findall("bndbox"):
            x1 = round(float(box.find("xmin").text))
            y1 = round(float(box.find("ymin").text))
            x2 = round(float(box.find("xmax").text))
            y2 = round(float(box.find("ymax").text))

            object_classes[i] = config.CLASSES.index(class_name)
            bboxes[i] = [x1, y1, x2, y2]

    return bboxes, object_classes

if __name__ == "__main__":
    img1, img2 = get_bboxes("dataset/VOC2012/Annotations/2007_000032.xml")
    print(img1.shape)
    print(img1.dtype)
