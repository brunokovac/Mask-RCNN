import os
import numpy as np
import math
import image_util
import xml_util

class Dataset():

    images = "/JPEGImages/"
    localization = "/Annotations/"
    recognition = "/SegmentationObject/"

    def __init__(self, path, data_list, batch_size=None):
        self.path = path
        self.batch_size = batch_size

        with open(path + data_list) as f:
            content = f.readlines()
        self.data_names = np.array([x.strip() for x in content])

        self.new_epoch = True
        self.current_batch = 0
        if batch_size:
            self.total_batches = math.ceil(len(self.data_names) // self.batch_size)
        return

    def next_batch_names(self):
        if self.new_epoch:
            np.random.shuffle(self.data_names)

        batch = self.data_names[self.current_batch * self.batch_size :
                                min((self.current_batch + 1) * self.batch_size, len(self.data_names))]

        if (self.current_batch + 1) * self.batch_size >= len(self.data_names):
            self.new_epoch = True
            self.current_batch = 0
        else:
            self.new_epoch = False
            self.current_batch += 1

        return batch

    def next_batch(self):
        img_names = self.next_batch_names()

        images = []
        boxes = []
        classes = []
        hws = []
        for i in range(len(img_names)):
            img_name = img_names[i]
            img_path = self.path + self.images + img_name
            xml_path = self.path + self.localization + img_name.replace("jpg", "xml")

            img, height, width = image_util.load_image(img_path)
            bboxes, object_classes = xml_util.get_bboxes(xml_path)
            images.append(img)
            hws.append([height, width])
            boxes.append(bboxes)
            classes.append(object_classes)

        return np.array(images), np.array(boxes), np.array(classes), np.array(hws)

    def load_all(self):
        np.random.shuffle(self.data_names)

        images = []
        boxes = []
        classes = []
        hws = []

        for i in range(len(self.data_names)):
            img_name = self.data_names[i]
            img_path = self.path + self.images + img_name
            xml_path = self.path + self.localization + img_name.replace("jpg", "xml")

            img, height, width = image_util.load_image(img_path)
            bboxes, object_classes = xml_util.get_bboxes(xml_path)
            images.append(img)
            hws.append([height, width])
            boxes.append(bboxes)
            classes.append(object_classes)

        return np.array(images), np.array(boxes), np.array(classes), np.array(hws)

if __name__ ==  "__main__":
    ds = Dataset("dataset/VOC2012", "/train_list.txt", 10)
    print(ds.total_batches)

