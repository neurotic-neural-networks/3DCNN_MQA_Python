import tensorflow as tf

def evalLoss(y_true, y_pred):
    
    GDT_TSmax = tf.reduce_max(y_true)
    scoreMin = tf.argmin(y_pred)
    GDT_TSargmin = tf.gather_nd(y_true, [scoreMin])

    return tf.abs(GDT_TSmax - GDT_TSargmin)