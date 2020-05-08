import rpn
import backbone
import dataset_util
import anchor_utils
import tensorflow as tf
import config
import mask_rcnn
import losses
import pdb

w1 = tf.Variable(tf.random.normal([4]))
print("0", w1)

checkpoint = tf.train.Checkpoint(net=w1, step=tf.Variable(1))
manager = tf.train.CheckpointManager(checkpoint, "ckpts2", max_to_keep=4)
if manager.latest_checkpoint:
    print("Restoring...", manager.latest_checkpoint)
    checkpoint.restore(manager.latest_checkpoint)

print("1", w1)

for epoch in range(5):
    w1.assign_add([1, 0, 1, 0])

    if epoch % 5 == 0:
        checkpoint.step.assign_add(1)
        manager.save()

