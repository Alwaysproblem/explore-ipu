#%%
import os
from statistics import mean
import time
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow_core.python.ipu as ipu
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
# from tensorflow.python.ipu import ipu_compiler, loops, ipu_infeed_queue, ipu_outfeed_queue, scopes
# from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
# from tensorflow.python.ipu.ops import application_compile_op
import yaml

os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --show_progress_bar=true --use_ipu_model'

np.random.seed(1991)

ops.disable_eager_execution()
tf.disable_v2_behavior()

def load_tf_graph(frozen_graph_filename, tag = tf.saved_model.tag_constants.SERVING):
        if not os.path.isdir(frozen_graph_filename):
            model_path = os.path.dirname(frozen_graph_filename)
        else:
            model_path = frozen_graph_filename

        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph = tf.saved_model.loader.load(sess, [tag] if tag else [], model_path)
            graph = tf.get_default_graph()
            return graph, meta_graph


def gen_data(graph):
    data = np.load("bs1_bps1.npz")
    feed_dict = {}

    StandardKvParser = graph.get_tensor_by_name("TensorDict/StandardKvParser:0")
    StandardKvParser_2 = graph.get_tensor_by_name("TensorDict/StandardKvParser_2:0")
    StandardKvParser_3 = graph.get_tensor_by_name("TensorDict/StandardKvParser_3:0")
    StandardKvParser_7 = graph.get_tensor_by_name("TensorDict/StandardKvParser_7:0")
    StandardKvParser_14 = graph.get_tensor_by_name("TensorDict/StandardKvParser_14:0")

    feed_dict[StandardKvParser] = np.expand_dims(data["tensordict_standardkvparser_0_arg"], axis = 0)
    feed_dict[StandardKvParser_2] = np.expand_dims(data["tensordict_standardkvparser_2_arg"], axis = 0)
    feed_dict[StandardKvParser_3] = np.expand_dims(data["tensordict_standardkvparser_3_arg"], axis = 0)
    feed_dict[StandardKvParser_7] = np.expand_dims(data["tensordict_standardkvparser_7_arg"], axis = 0)
    feed_dict[StandardKvParser_14] = np.expand_dims(data["tensordict_standardkvparser_14_arg"], axis = 0)

    return feed_dict


def run_model(model_path, output_op_names):
    graph, meta = load_tf_graph(model_path)
    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )

    out_names_pl = [ graph.get_tensor_by_name(o_name) for o_name in output_op_names]

    with tf.Session(graph=graph, config=sess_cfg) as session:
        feed_dict = gen_data(graph)
        results = session.run(out_names_pl, feed_dict=feed_dict)

    tf.reset_default_graph()
    return results

def check_same(s, f):
    return list(map(lambda x, y: np.all(x == y), s, f))

if __name__ == "__main__":
    with open("qtc-trans/outop.yml") as ff:
        config_yaml = yaml.load(ff, Loader=yaml.FullLoader)

    model_stand = config_yaml["modelURL"]["standard"]
    model_needfix = config_yaml["modelURL"]["needfix"]

    output_op_list = config_yaml["outputNode"]

    standard_output = run_model(model_stand, output_op_list)
    needfix_output = run_model(model_needfix, output_op_list)

    print(standard_output)
    print(needfix_output)

    print(f"the check same: {check_same(standard_output, needfix_output)}")
