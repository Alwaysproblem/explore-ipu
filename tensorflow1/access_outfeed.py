from threading import Thread

from tensorflow.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
from tensorflow.python import keras
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ds = tf.data.Dataset.from_tensor(tf.constant(1.0, shape=[2, 20]))
ds = ds.repeat()

infeed_queue = ipu_feed_queue