import tensorflow as tf

def rpn_object_loss(object_gt, object_predicted):
    indices = tf.where(object_gt != 0)

    gt_selected = tf.gather_nd(object_gt, indices)
    gt_selected = tf.cast(gt_selected == 1, tf.int32)
    pred_selected = tf.gather_nd(object_predicted, indices)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_selected, pred_selected)
    return loss

def rpn_bbox_loss(object_gt, bbox_gt, bbox_predicted):
    indices = tf.where(object_gt == 1)

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
    positive_indices = tf.cast(tf.where(gt_classes > 0), "int32")
    classes_indices = tf.gather_nd(gt_classes, positive_indices)
    all_indices = tf.concat([positive_indices, tf.transpose([classes_indices])], axis=1)

    selected_gt_bboxes = tf.gather_nd(gt_bboxes, positive_indices)
    selected_predicted_bboxes = tf.gather_nd(predicted_bboxes, all_indices)

    diff = tf.abs(selected_gt_bboxes - selected_predicted_bboxes)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return tf.reduce_mean(loss) if loss.shape[0] != 0 else 0.0

def mask_rcnn_mask_loss(gt_classes, gt_masks, predicted_masks):
    positive_indices = tf.cast(tf.where(gt_classes > 0), "int32")
    classes_indices = tf.gather_nd(gt_classes, positive_indices)
    all_indices = tf.concat([positive_indices, tf.transpose([classes_indices])], axis=1)

    selected_gt_masks = tf.gather_nd(gt_masks, positive_indices)
    selected_predicted_masks = tf.gather_nd(predicted_masks, all_indices)

    losses = tf.keras.losses.BinaryCrossentropy(reduction="none")(selected_gt_masks, selected_predicted_masks)
    selected_losses = tf.gather_nd(losses, tf.where(tf.greater(losses, 0)))
    return tf.reduce_mean(selected_losses) if selected_losses.shape[0] != 0 else 0.0
