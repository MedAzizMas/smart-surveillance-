import tensorflow as tf

def iou_metric(y_true, y_pred):
    """Calculate Intersection over Union (IoU) metric"""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7) 