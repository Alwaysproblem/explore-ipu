import numpy as np
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope, ipu_shard
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops 
import tensorflow._api.v1.compat.v1 as tf
from tensorflow.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)

NUM_IPUS = 4

cfg = create_ipu_config(profiling=True, use_poplar_text_report=True,)
cfg = set_ipu_model_options(cfg, compile_ipu_code=True)
cfg = auto_select_ipus(cfg, NUM_IPUS)
configure_ipu_system(cfg)

with tf.device("cpu"):
    pa = tf.placeholder(np.float32, [2], name = "a")
    pb = tf.placeholder(np.float32, [2], name = "d")
    pc = tf.placeholder(np.float32, [2], name = "c")

with tf.device("cpu"):
    report = gen_ipu_ops.ipu_event_trace()

def shard_graph(pa, pb, pc):
    with ipu_shard(0):
        o1 = pa + pb
    
    with ipu_shard(1):
        o2 = pa + pc
    
    with ipu_shard(2):
        o3 = pb + pc
    
    with ipu_shard(3):
        out = o1 + o2 + o3
    
    return out

with ipu_scope("/device:IPU:0"):
    res = ipu.ipu_compiler.compile(shard_graph, [pa, pb, pc])


config=tf.ConfigProto(log_device_placement=True)
with tf.Session(config=config) as sess:
    res = sess.run(res, feed_dict={
        pa: [1., 1.],
        pb: [0., 1.],
        pc: [1., 5.],
    })

    print(res)