import numpy as np


# from tensorflow.python import ipu
# from tensorflow.python.ipu.scopes import ipu_scope
# import tensorflow.compat.v1 as tf

# import tensorflow # to avoid import error
from tensorflow import TensorArray # to avoid import error
import tensorflow_core._api.v1.compat.v1 as tf
from tensorflow_core.python import ipu
from tensorflow_core.python.ipu.scopes import ipu_scope
from tensorflow_core.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)

tf.disable_v2_behavior()

cfg = create_ipu_config(profiling=True, use_poplar_text_report=True) 
cfg = set_ipu_model_options(cfg, compile_ipu_code=False)
cfg = auto_select_ipus(cfg, 1) 
configure_ipu_system(cfg)

with tf.device("cpu"):
    pa = tf.placeholder(np.float32, [2], name = "a")
    pb = tf.placeholder(np.float32, [2], name = "d")
    pc = tf.placeholder(np.float32, [2], name = "c")


def basic_graph(pa, pb, pc):
    o1 = pa + pb * pc
    o2 = pa + pc

    simple_graph_output = o1 + o2
    return simple_graph_output

with tf.device("/device:IPU:0"):
    result = basic_graph(pa, pb, pc)


with tf.Session() as sess:
    result = sess.run(result, feed_dict={
        pa: [1., 1.],
        pb: [0., 1.],
        pc: [1., 5.],
    })

print(result)