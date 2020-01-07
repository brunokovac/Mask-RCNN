import xml.etree.ElementTree as ET
import numpy as np

classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def get_bboxes(path):
    tree = ET.parse(path)
    root = tree.getroot()

    bboxes = []
    object_classes = []

    for boxes in root.iter('object'):

        class_name = boxes.find("name").text

        for box in boxes.findall("bndbox"):
            x1 = int(box.find("xmin").text)
            y1 = int(box.find("ymin").text)
            x2 = int(box.find("xmax").text)
            y2 = int(box.find("ymax").text)

            object_classes.append(classes.index(class_name))
            bboxes.append([x1, y1, x2, y2])

    return np.array(bboxes), np.array(object_classes)

if __name__ == "__main__":
    img1, img2 = get_bboxes("dataset/VOC2012/Annotations/2007_000032.xml")
    print(img1.shape)
    print(img1.dtype)
