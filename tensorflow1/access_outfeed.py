from threading import Thread

import tensorflow

# from tensorflow.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
# import tensorflow.compat.v1 as tf

import tensorflow_core._api.v1.compat.v1 as tf
# from tensorflow_core.python import ipu
from tensorflow_core.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
from tensorflow_core.python.ipu.scopes import ipu_scope
from tensorflow_core.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)
from tensorflow_core.python import keras

tf.disable_v2_behavior()

ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[2, 20]))
ds = ds.repeat()

infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds, feed_name = "infeed", prefetch_depth = 3)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name = "outfeed")


NUM_ITERATIONS = 100

def inference_step(image):
    partial = keras.layers.Dense(256, activation=tf.nn.relu)(image)
    partial = keras.layers.Dense(128, activation=tf.nn.relu)(partial)
    logits = keras.layers.Dense(10)(partial)
    classes = tf.argmax(input=logits, axis=1, output_type=tf.dtypes.int32)
    outfeed = outfeed_queue.enqueue(classes)
    return outfeed


def inference_loop():
    r = loops.repeat(NUM_ITERATIONS, inference_step, inputs=[], infeed_queue = infeed_queue)
    return r

with tf.device("/device:IPU:0"):
    run_loop = ipu_compiler.compile(inference_loop, inputs=[])

dequeue_outfeed = outfeed_queue.dequeue()


config = create_ipu_config(profiling=True, use_poplar_text_report=False)
config = auto_select_ipus(config, 1)
configure_ipu_system(config=config)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(infeed_queue.initializer)

    def dequeue():
        counter = 0

        while counter < NUM_ITERATIONS:
            r = sess.run(dequeue_outfeed)

            if r.size:
                for t in r:
                    print(f"Step: {counter}, class: {r}")
                    counter += 1
    sess.run(run_loop)

    dequeue_tread = Thread(target=dequeue)
    dequeue_tread.start()

    dequeue_tread.join()