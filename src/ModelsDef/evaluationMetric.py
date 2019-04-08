import keras.backend as K

def evaluationLoss(y_true, y_pred):

    GDT_TSmax = K.tf.reduce_max(y_true, axis=-1)
    scoreMin = K.tf.argmin(y_pred, axis=-1)
    temp = K.tf.cast(K.tf.range(0, K.tf.shape(y_pred)[0]), K.tf.int64)
    indices = K.tf.stack([temp, scoreMin], axis=1)
    GDT_TSargmin = K.tf.gather_nd(y_true,indices)

    return K.tf.abs(GDT_TSmax - GDT_TSargmin)