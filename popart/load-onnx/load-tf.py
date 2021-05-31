import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.saved_model import tag_constants
from tensorflow.python.ipu import utils
from tensorflow.python import ipu

np.random.seed(1)

# Builds ipu_options
config = utils.create_ipu_config()

config = utils.auto_select_ipus(config, 2)

config = utils.set_convolution_options(config, {
    "availableMemoryProportion": str(0.4),
    "partialsType": "half"
})
config = utils.set_matmul_options(config, {
    "availableMemoryProportion": str(0.4),
    "partialsType": "half",
})

ipu.utils.configure_ipu_system(config)

restored_graph = tf.Graph()
with restored_graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            'qtc35-tf-ipu',
        )
        StandardKvParser_4 = restored_graph.get_tensor_by_name('TensorDict/StandardKvParser_4:0')
        StandardKvParser_1 = restored_graph.get_tensor_by_name('TensorDict/StandardKvParser_1:0')
        StandardKvParser_6 = restored_graph.get_tensor_by_name('TensorDict/StandardKvParser_6:0')
        StandardKvParser_8 = restored_graph.get_tensor_by_name('TensorDict/StandardKvParser_8:0')
        prediction = restored_graph.get_tensor_by_name('concat:0')

        feed_dict = {
            StandardKvParser_4: np.random.randint(12, size=[1, 512]).astype(np.int64),
            StandardKvParser_1: np.random.randint(12, size=[1, 64]).astype(np.int64),
            StandardKvParser_6: np.random.randint(12, size=[1, 512]).astype(np.int64),
            StandardKvParser_8: np.random.randint(12, size=[1, 64]).astype(np.int64),
            # prediction: another_value,
        }

        prob = sess.run(prediction, feed_dict=feed_dict)
        print(prob)