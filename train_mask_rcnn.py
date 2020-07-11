import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import mask_rcnn
import losses
import numpy as np
import sys
import metrics

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as gt:
        gt_boxes, gt_classes, gt_masks, img_sizes, gt_rpn_classes, gt_rpn_bbox_deltas = labels

        rpn_fg_bg_softmaxes, rpn_bbox_deltas, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, mask_rcnn_masks, proposals = model(data, training=True)

        rpn_object_loss = losses.rpn_object_loss(gt_rpn_classes, rpn_fg_bg_softmaxes)
        rpn_bbox_loss = losses.rpn_bbox_loss(gt_rpn_classes, gt_rpn_bbox_deltas, rpn_bbox_deltas)

        mask_rcnn_gt_proposals, mask_rcnn_predicted_classes, mask_rcnn_predicted_bbox_deltas, mask_rcnn_predicted_masks, \
        mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_gt_masks = \
            mask_rcnn.generate_mask_rcnn_labels(proposals, mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, mask_rcnn_masks, gt_classes, gt_boxes, gt_masks)
        mask_rcnn_class_loss = losses.mask_rcnn_class_loss(mask_rcnn_gt_classes, mask_rcnn_predicted_classes)
        mask_rcnn_bbox_loss = losses.mask_rcnn_bbox_loss(mask_rcnn_gt_classes, mask_rcnn_gt_deltas, mask_rcnn_predicted_bbox_deltas)
        mask_rcnn_mask_loss = losses.mask_rcnn_mask_loss(mask_rcnn_gt_classes, mask_rcnn_gt_masks, mask_rcnn_predicted_masks)

        loss = rpn_object_loss + rpn_bbox_loss + mask_rcnn_class_loss + mask_rcnn_bbox_loss + mask_rcnn_mask_loss

    grads = gt.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rpn_object_loss, rpn_bbox_loss, mask_rcnn_class_loss, mask_rcnn_bbox_loss, mask_rcnn_mask_loss

def calculate_average_ap_on_batch(images, gt_boxes, gt_classes, gt_masks, pred_boxes, pred_classes, pred_classes_scores, pred_masks):
    mask_sum = 0
    box_sum = 0

    for i in range(len(images)):
        mask_ap, box_ap = metrics.compute_ap(gt_boxes[i], gt_classes[i], gt_masks[i], pred_boxes[i], pred_classes[i], pred_classes_scores[i], pred_masks[i])
        mask_sum += mask_ap
        box_sum += box_ap

    return mask_sum, box_sum

def calculate_map(dataset, model):
    mask_sum = 0
    box_sum = 0

    for _ in range(dataset.total_batches):
        images, gt_boxes, gt_classes, gt_masks, img_sizes = dataset.next_batch()

        data = [images, img_sizes]
        pred_boxes, pred_classes_scores, pred_classes, pred_masks, rpn_fg_bg_softmaxes, rpn_bbox_deltas, \
        mask_rcnn_classes_softmax, mask_rcnn_bbox_deltas, mask_rcnn_masks, proposals = model(data, training=False)

        mask_ap, box_ap = calculate_average_ap_on_batch(images, gt_boxes, gt_classes, gt_masks, pred_boxes, pred_classes, pred_classes_scores, mask_rcnn_masks)
        mask_sum += mask_ap
        box_sum += box_ap

    return mask_sum / len(dataset.data_names), box_sum / len(dataset.data_names)

def train(num_epochs, optimizer, anchors, train_dataset, td_map, vd_map):
    max_map = float("inf")
    bigger_map_in_row = 0
    for epoch in range(1, num_epochs + 1):
        print("Epoch", epoch)
        epoch_losses = np.zeros(5)
        for i in range(1, train_dataset.total_batches + 1):
            images, gt_boxes, gt_classes, gt_masks, img_sizes = train_dataset.next_batch()
            gt_rpn_classes, gt_rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)
            l1, l2, l3, l4, l5 = train_step(model, optimizer, [images, img_sizes],
                                           [gt_boxes, gt_classes, gt_masks, img_sizes, gt_rpn_classes, gt_rpn_bbox_deltas])

            l1, l2, l3, l4, l5 = tf.keras.backend.eval(l1), tf.keras.backend.eval(l2), tf.keras.backend.eval(l3), \
                                 tf.keras.backend.eval(l4), tf.keras.backend.eval(l5)
            epoch_losses += [l1, l2, l3, l4, l5]

            if i % 10 == 0:
                print("Iter {}: rpn_cls_loss={}, rpn_bbox_loss={}, mask_cls_loss={}, mask_bbox_loss={}, mask_mask_loss={}".format(i, l1, l2, l3, l4, l5))

        epoch_losses /= train_dataset.total_batches
        print("*" * 50)
        print("Epoch {}: rpn_cls_loss={}, rpn_bbox_loss={}, mask_cls_loss={}, mask_bbox_loss={}, mask_mask_loss={}".format(epoch, *epoch_losses))
        print("*" * 50)
        with open(config.TRAIN_LOSSES_FILE, "a+") as f1:
            f1.write("{} {} {} {} {}\n".format(*epoch_losses))

        if epoch % 10 == 0:
            checkpoint.step.assign_add(1)
            manager.save()

            train_mask_map, train_box_map = calculate_map(td_map, model)
            print("Train mAP: mask", train_mask_map, "bbox", train_box_map)

            valid_mask_map, valid_box_map = calculate_map(vd_map, model)
            print("Valid mAP: mask", valid_mask_map, "bbox", valid_box_map)

            if valid_mask_map < max_map:
                max_map = valid_mask_map
                bigger_map_in_row = 0
            else:
                bigger_map_in_row += 1

                if bigger_map_in_row == 10:
                    print("{}. bigger map in row, exiting".format(bigger_map_in_row))
                    sys.exit(0)

if __name__ == "__main__":
    #tf.executing_eagerly()
    anchors = anchor_utils.get_all_anchors(config.IMAGE_SIZE, config.ANCHOR_SCALES, config.ANCHOR_RATIOS)

    backbone2 = backbone.Resnet34_FPN()
    rpn2 = rpn.RPN(backbone2, 3)
    model = mask_rcnn.Mask_RCNN(rpn2, anchors, len(config.CLASSES))

    optimizer = tf.keras.optimizers.SGD(lr=0.004, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model, step=tf.Variable(1))
    manager = tf.train.CheckpointManager(checkpoint, config.WEIGHTS_DIR, max_to_keep=4)

    train_dataset = dataset_util.VOC2012_Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 12)
    td_map = dataset_util.VOC2012_Dataset("DATASET/VOC2012/VOC2012", "/train_list.txt", 4)
    vd_map = dataset_util.VOC2012_Dataset("DATASET/VOC2012/VOC2012", "/valid_list.txt", 4)

    '''train_dataset = dataset_util.AOLP_Dataset("DATASET/AOLP", "/train_list.txt", 10)
    td_map = dataset_util.AOLP_Dataset("DATASET/AOLP", "/train_list.txt", 4)
    vd_map = dataset_util.AOLP_Dataset("DATASET/AOLP", "/valid_list.txt", 4)'''

    if manager.latest_checkpoint:
        print("Restoring...", manager.latest_checkpoint)
        images, gt_boxes, gt_classes, gt_masks, img_sizes = train_dataset.next_batch()
        rpn_classes, rpn_bbox_deltas = anchor_utils.get_rpn_classes_and_bbox_deltas(len(images), anchors, gt_boxes)
        l1, l2, l3, l4, l5 = train_step(model, optimizer, [images, img_sizes], [gt_boxes, gt_classes, gt_masks, img_sizes, rpn_classes, rpn_bbox_deltas])
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()

    mask_map, box_map = calculate_map(td_map, model)
    print("Train mAP: mask", mask_map, "bbox", box_map)
    mask_map, box_map = calculate_map(vd_map, model)
    print("Valid mAP: mask", mask_map, "bbox", box_map)

    train(500, optimizer, anchors, train_dataset, td_map, vd_map)