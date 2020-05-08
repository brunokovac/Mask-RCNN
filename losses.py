import tensorflow as tf
import config

def rpn_object_loss(object_gt, object_predicted):
    object_gt = tf.squeeze(object_gt, axis=-1)
    indices = tf.where(tf.not_equal(object_gt, 0))

    gt_selected = tf.gather_nd(object_gt, indices)
    gt_selected = tf.cast(tf.equal(gt_selected, 1), tf.int32)
    pred_selected = tf.gather_nd(object_predicted, indices)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_selected, pred_selected)
    return loss

def rpn_bbox_loss(object_gt, bbox_gt, bbox_predicted):
    object_gt = tf.squeeze(object_gt, axis=-1)
    indices = tf.where(tf.equal(object_gt, 1))

    gt_bbox_selected = tf.gather_nd(bbox_gt, indices)
    pred_bbox_selected = tf.gather_nd(bbox_predicted, indices)

    diff = tf.abs(tf.cast(gt_bbox_selected, "float32") - pred_bbox_selected)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_mean(loss)
    return loss

def mask_rcnn_class_loss(gt_classes, predicted_classes_softmax):
    losses = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")(gt_classes, predicted_classes_softmax)
    selected_losses = tf.gather_nd(losses, tf.where(tf.greater(losses, 0)))
    return tf.reduce_mean(selected_losses)

def mask_rcnn_bbox_loss(gt_classes, gt_bboxes, predicted_bboxes):
    positive_indices = tf.where(gt_classes > 0)
    gt_classes = tf.gather_nd(gt_classes, positive_indices)

    gt_bboxes_selected = tf.gather_nd(gt_bboxes, positive_indices)
    bboxes_indices = tf.concat([tf.cast(positive_indices, "int32"), tf.transpose([gt_classes])], axis=1)
    pred_bboxes_selected = tf.gather_nd(predicted_bboxes, bboxes_indices)

    diff = tf.abs(gt_bboxes_selected - pred_bboxes_selected)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_mean(loss)
    return loss