import numpy as np
import sys
import os
# TF_POPLAR_FLAGS = os.environ.get("TF_POPLAR_FLAGS", None)
# os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model" if TF_POPLAR_FLAGS is None else TF_POPLAR_FLAGS + ' --use_ipu_model'

# import tensorflow # to avoid import error
from tensorflow import TensorArray # to avoid import error
import tensorflow_core._api.v1.compat.v1 as tf
from tensorflow_core.python import ipu
from tensorflow_core.python.ipu.scopes import ipu_scope
from tensorflow_core.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)

tf.disable_v2_behavior()

cfg = create_ipu_config(profiling=True, use_poplar_text_report=True) 
cfg = set_ipu_model_options(cfg, compile_ipu_code=True)
cfg = auto_select_ipus(cfg, 2) 
configure_ipu_system(cfg)

inputs_shape_and_dtype = ('a', (2, ), np.float32), ('b', (5, ), np.float32) # (shape, dtype)
dataset = tf.data.Dataset.from_tensors(
        tuple([np.random.randint(10, size=shape).astype(dtype)
        for _, shape, dtype in inputs_shape_and_dtype])
    )
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=2, drop_remainder=True)
dataset = dataset.shuffle(100)
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name=f"infeed")

def stage_0(input_0, input_1):
    return tf.layers.dense(input_0, units=2, activation=tf.nn.relu,
                kernel_initializer=tf.initializers.truncated_normal(), 
                bias_initializer=tf.initializers.ones()), input_1

def stage_1(input_0, input_1):
    return tf.layers.dense(input_0, units=2, activation=tf.nn.relu,
                kernel_initializer=tf.initializers.truncated_normal(), 
                bias_initializer=tf.initializers.ones()), input_1

def stage_2(input_0, input_1):
    return input_0, tf.layers.dense(input_1, units=2, activation=tf.nn.relu,
                kernel_initializer=tf.initializers.truncated_normal(), 
                bias_initializer=tf.initializers.ones())

def stage_3(input_0, input_1):
    return tf.concat([input_0, input_1], axis = 1)

outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")
computational_stages = [stage_0, stage_1, stage_2, stage_3]
device_mapping = [0, 1, 0, 1]


pipeline_op = ipu.pipelining_ops.pipeline(
            computational_stages=computational_stages,
            device_mapping=device_mapping,
            gradient_accumulation_count=4,
            repeat_count=1,
            inputs=[],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            name='pipeline_op')

with tf.device("/device:IPU:0"):
    r = ipu.ipu_compiler.compile( lambda: pipeline_op, inputs=[])


if compile_only = True:
    with tf.Session() as sess:
        compile_only = True
        try:
            sess.run(tf.global_variables_initializer())
        except tf.errors.OpError as e:
            if compile_only and 'compilation only' in e.message:
                print("Validation graph successfully compiled")
                print("Exiting...")
                sys.exit(0)
            raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)
else:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(infeed_queue.initializer)
        sess.run(r)
        print(sess.run(outfeed_queue.dequeue()))
