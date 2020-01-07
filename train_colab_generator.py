import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import numpy as np

def data_generator(dataset, anchors):
    while True:
        batch_images, batch_boxes, batch_classes = dataset.next_batch()
        batch_classes, batch_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(batch_images), anchors, batch_boxes)
        print("getting new batch")
        yield [batch_images, batch_classes, batch_bbox_deltas], [batch_classes, batch_bbox_deltas]

def one_hot(a):
    a = np.array(a, dtype=np.int32)
    b = np.zeros((a.size, a.max() + 1), dtype="float32")
    b[np.arange(a.size), a] = 1
    return b

model = rpn.RPN(backbone.Resnet34_FPN(), 3).model()
model.compile(opt=tf.keras.optimizers.Adam(0.001), metrics=["accuracy"])

anchors = anchor_utils.get_all_anchors((512, 512), [32, 64, 128, 256, 512], [(1, 1), (1, 2), (2, 1)])

ds1 = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 64)
train_generator = data_generator(ds1, anchors)

ds2 = dataset_util.Dataset("DATASET/VOC2012/VOC2012", "/valid_list.txt", 64)
valid_generator = data_generator(ds2, anchors)

model.fit_generator(generator=train_generator, validation_data=valid_generator,
                    epochs=5, steps_per_epoch=ds1.total_batches, validation_steps=ds2.total_batches)

