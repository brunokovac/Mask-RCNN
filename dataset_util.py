import numpy as np
import math
import image_util
import xml_util
import config
import skimage.transform as st
import os.path as path_check

class VOC2012_Dataset():

    images = "/JPEGImages/"
    localization = "/Annotations/"
    segmentation = "/SegmentationObject/"

    def __init__(self, path, data_list, batch_size=None):
        self.path = path
        self.batch_size = batch_size

        with open(path + data_list) as f:
            content = f.readlines()
        self.data_names = np.array([x.strip() for x in content])

        self.new_epoch = True
        self.current_batch = 0
        if batch_size:
            self.total_batches = math.ceil(len(self.data_names) / self.batch_size)
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
        masks = []
        hws = []
        for i in range(len(img_names)):
            img_name = img_names[i]
            img_path = self.path + self.images + img_name + ".jpg"
            xml_path = self.path + self.localization + img_name + ".xml"

            img, height, width = image_util.load_image(img_path)
            images.append(img)
            hws.append([height, width])

            if not path_check.exists(xml_path):
                continue

            bboxes, object_classes = xml_util.get_bboxes(xml_path)
            boxes.append(bboxes)
            classes.append(object_classes)

            mask_img_path = self.path + self.segmentation + img_name + ".png"
            masks_image = image_util.load_mask(mask_img_path)
            masks_i = np.zeros((config.MAX_OBJECTS_PER_IMAGE, *config.MASK_SHAPE))
            for i in range(len(bboxes)):
                box_i = bboxes[i]
                if np.sum(box_i) == 0:
                    break

                mask = masks_image[box_i[1]:box_i[3], box_i[0]:box_i[2]]
                mask = np.where(mask == (i + 1), 1, 0)
                resized_mask = st.resize(mask, config.MASK_SHAPE, preserve_range=True, anti_aliasing=False)
                masks_i[i] = resized_mask

            masks.append(masks_i)

        return np.array(images), np.array(boxes), np.array(classes), np.array(masks), np.array(hws)

class AOLP_Dataset():

    images = "/Image/"
    localization = "/groundtruth_localization/"

    def __init__(self, path, data_list, batch_size=None):
        self.path = path
        self.batch_size = batch_size

        with open(path + data_list) as f:
            content = f.readlines()
        self.data_names = np.array([x.strip() for x in content])

        self.new_epoch = True
        self.current_batch = 0
        if batch_size:
            self.total_batches = math.ceil(len(self.data_names) / self.batch_size)
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
        masks = []
        hws = []
        for i in range(len(img_names)):
            img_name = img_names[i]
            img_path = self.path + self.images + img_name + ".jpg"
            boxes_path = self.path + self.localization + img_name + ".txt"

            img, height, width = image_util.load_image(img_path)

            if not path_check.exists(boxes_path):
                continue

            gt_boxes = np.reshape(np.loadtxt(boxes_path, dtype="int32"), [-1, 4])
            bboxes = np.zeros((config.MAX_OBJECTS_PER_IMAGE, 4))
            for j in range(len(gt_boxes)):
                if gt_boxes[j][0] > gt_boxes[j][2]:
                    tmp = gt_boxes[j][2]
                    gt_boxes[j][2] = gt_boxes[j][0]
                    gt_boxes[j][0] = tmp
                if gt_boxes[j][1] > gt_boxes[j][3]:
                    tmp = gt_boxes[j][3]
                    gt_boxes[j][3] = gt_boxes[j][1]
                    gt_boxes[j][1] = tmp
            bboxes[:len(gt_boxes)] = gt_boxes

            object_classes = np.zeros(config.MAX_OBJECTS_PER_IMAGE)
            object_classes[:len(gt_boxes)] = 1

            images.append(img)
            hws.append([height, width])
            boxes.append(bboxes)
            classes.append(object_classes)

            masks_i = np.zeros((config.MAX_OBJECTS_PER_IMAGE, *config.MASK_SHAPE))
            masks_i[:len(gt_boxes)] = np.ones(config.MASK_SHAPE)
            masks.append(masks_i)

        return np.array(images), np.array(boxes), np.array(classes), np.array(masks), np.array(hws)

if __name__ ==  "__main__":
    ds = VOC2012_Dataset("dataset/VOC2012", "/train_list.txt", 1)
    ds.next_batch()

