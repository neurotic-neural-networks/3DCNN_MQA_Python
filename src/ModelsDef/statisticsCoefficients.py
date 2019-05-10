import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import spearmanr, kendalltau

def Pearson(y_true, y_pred):
    return tfp.stats.correlation(y_pred, y_true, event_axis=None)

def Spearman(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32) )

def Kendall(y_true, y_pred):
    return (tf.py_function(kendalltau, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32) )

    
