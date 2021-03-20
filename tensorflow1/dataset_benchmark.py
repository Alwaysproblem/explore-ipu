import tensorflow
import json
# from tensorflow.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
# import tensorflow.compat.v1 as tf

import tensorflow_core._api.v1.compat.v1 as tf
from tensorflow_core.python import ipu
from tensorflow_core.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
from tensorflow_core.python.ipu.scopes import ipu_scope
from tensorflow_core.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)
from tensorflow_core.python import keras

tf.disable_v2_behavior()

ds = tf.data.Dataset.from_tensors(tf.random.normal(shape=[2, 20]))
ds = ds.repeat()
ds = ds.prefetch(1000)


benchmark_op = ipu.dataset_benchmark.dataset_benchmark(ds, 10, 512)

with tf.Session() as sess:
    json_string = sess.run(benchmark_op)
    print(json_string[0].decode())