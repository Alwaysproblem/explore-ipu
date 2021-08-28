#%%
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import types_pb2, attr_value_pb2

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants
import shutil
import numpy as np
#%%
model_for_trans = 'qtcv211-tf-ipu'
model_save_after = "qtcv211-tf-ipu-int32"
# model_for_trans = 'ipu_only_ipugelu_fp16'
# model_save_after = "ipu_only_ipugelu_fp16_int32"

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], model_for_trans)
    graph = tf.get_default_graph()

    signature = meta_graph_def.signature_def

graph_def = graph.as_graph_def()
new_graph_def = tf.Graph().as_graph_def()
new_graph_def.versions.CopyFrom(graph_def.versions)

name_pbnode_dict = {node.name: node for node in graph_def.node} 

#%%
def analyze_pb_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)

    return inputs, outputs

def change_T(new_node, *attr):
    for a in attr:
        if a in ("T", 'DstT', 'Tindices', 'TI'):
            new_node.attr[a].type = types_pb2.DT_INT32
        elif a in ("dtype", ):
            new_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_INT32))
        elif a in ("value", ):
            values = tf.make_ndarray(new_node.attr['value'].tensor).astype(np.int32)
            new_node.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(values, dtype=tf.int32))
            # new_node.attr['value'].tensor.dtype = types_pb2.DT_INT32


#%%
operationdic = {
    'TensorDict/StandardKvParser': ['dtype'],
    'TensorDict/StandardKvParser_3': ['dtype'],
    'TensorDict/StandardKvParser_7': ['dtype'],
    'TensorDict/StandardKvParser_14': ['dtype'],
    'TensorDict/StandardKvParser_2': ['dtype'],
    'reload_query/bert/embeddings/Shape': ['T'],
    'reload_query/zeros_like': ['T'],
    'reload_query/ones_like': ['T'],
    'reload_query/mul': ['T'],
    'reload_query/mul_1': ['T'],
    'reload_query/mul_2': ['T'],
    'reload_query/ToInt64': ['DstT'],
    'reload_query/add_1': ['T'],
    'reload_query/bert/embeddings/embedding_lookup': ['Tindices'],
    'reload_query/ToInt64_1': ['DstT'],
    'reload_query/bert/embeddings/ExpandDims': ['T'],
    'reload_query/strided_slice': ['T'],
    'reload_query/strided_slice_1': ['T'],
    'reload_query/strided_slice_2': ['T'],
    'reload_query/concat': ['T'],
    'reload_query/ones_like/Shape': ['T'],
    'reload_query/Equal': ['T'],
    'reload_query/sub': ['T'],
    'reload_query/mul_3': ['T'],
    'reload_query/bert/embeddings/one_hot': ['TI'],
    'reload_query/NotEqual': ['T'],
    'reload_query/sub/x': ['dtype', "value"],
    'reload_query/ones_like/Const': ['dtype', "value"],
    'reload_query/mul_2/y': ['dtype', "value"],
    'reload_query/Equal/y': ['dtype', "value"],
    'reload_query/NotEqual/y': ['dtype', "value"],
    'reload_query/mul/y': ['dtype', "value"],
    "reload_query/bert/embeddings/Reshape_1": ["T"],
    "reload_query/bert/encoder/Shape": ["T"],
    "reload_qt/bert/embeddings/ExpandDims": ["T"],
    "reload_qt/bert/embeddings/Shape": ["T"],
    'reload_qt/bert/embeddings/one_hot': ['TI'],
    'reload_qt/bert/embeddings/embedding_lookup': ['Tindices'],
    "reload_qt/NotEqual": ["T"],
    "reload_qt/bert/encoder/Shape": ["T"],
    "reload_qt/semantics_cnn/ExpandDims": ["T"],
    "reload_qt/semantics_cnn/strided_slice_1": ["T"],
    "reload_qt/semantics_cnn/Greater": ["T"],
    "reload_qt/semantics_cnn/ExpandDims_1": ["T"],
    "reload_qt/semantics_cnn/strided_slice_2": ["T"],
    "reload_qt/semantics_cnn/Equal": ["T"],
    'reload_qt/semantics_cnn/Equal_1/y': ['dtype', "value"],
    'reload_qt/semantics_cnn/Equal/y': ['dtype', "value"],
    'reload_qt/semantics_cnn/Greater/y': ['dtype', "value"],
    "reload_qt/semantics_cnn/ExpandDims_2": ["T"],
    "reload_qt/semantics_cnn/strided_slice_4": ["T"],
    "reload_qt/semantics_cnn/Equal_1": ["T"],
    "reload_qt/bert/embeddings/Reshape_1": ["T"],
    "reload_qt/bert/embeddings/one_hot": ["TI"],
    'reload_qt/NotEqual/y': ['dtype', "value"],
    'reload_qt/NotEqual': ["T"],
    "reload/bert/embeddings/ExpandDims": ["T"],
    "reload/bert/embeddings/Shape": ["T"],
    'reload/bert/embeddings/one_hot': ['TI'],
    'reload/bert/embeddings/embedding_lookup': ['Tindices'],
    "reload/bert/embeddings/Reshape_1": ["T"],
    "reload/NotEqual": ["T"],
    "reload/bert/encoder/Shape": ["T"],
    "reload/NotEqual/y": ['dtype', "value"],
    "reload/semantics_cnn/ExpandDims": ["T"],
    "reload/semantics_cnn/ExpandDims_1": ["T"],
    "reload/semantics_cnn/ExpandDims_2": ["T"],
    "reload/semantics_cnn/strided_slice_1": ["T"],
    "reload/semantics_cnn/strided_slice_2": ["T"],
    "reload/semantics_cnn/strided_slice_4": ["T"],
    'reload/semantics_cnn/Greater_1/y': ['dtype', "value"],
    'reload/semantics_cnn/Greater_1': ["T"],
    'reload/semantics_cnn/Equal_1/y': ['dtype', "value"],
    'reload/semantics_cnn/Equal_1': ["T"],
    'reload/semantics_cnn/Equal/y': ['dtype', "value"],
    'reload/semantics_cnn/Equal': ["T"],
    'reload/semantics_cnn/Greater/y': ['dtype', "value"],
    "reload/semantics_cnn/Greater": ["T"],
}

#%%
for node in graph_def.node:
    if 'T' in node.attr.keys() and node.attr['T'].type == types_pb2.DT_INT64:
        node.attr['T'].type = types_pb2.DT_INT32
    if node.name in operationdic:
        change_T(node, *operationdic[node.name])

#%%
name_pbnode_dict = {node.name: node for node in graph_def.node}

for node in graph_def.node:
    added_node = new_graph_def.node.add()
    added_node.CopyFrom(name_pbnode_dict[node.name])

#%%
with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(add_shapes=True), ["concat"])

shutil.rmtree(model_save_after, ignore_errors=True)
saved_model_builder = builder.SavedModelBuilder(model_save_after)
with tf.Graph().as_default():
    tf.import_graph_def(constant_graph, name="")
    # We don't use any specific converter here.
    with tf.Session() as sess:
        saved_model_builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map=signature)

saved_model_builder.save()

#%%
# with tf.Session(graph=tf.Graph()) as sess:
#     meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'qtc-int32-tf')
#     graph = tf.get_default_graph()

#     signature = meta_graph_def.signature_def
#%%
# Write the transformed graphdef as SavedModel.
# saved_model_builder = builder.SavedModelBuilder(output_saved_model_dir)
# with tf.Graph().as_default():
#     tf.import_graph_def(self._converted_graph_def, name="")
#     # We don't use any specific converter here.
#     with session.Session() as sess:
#     saved_model_builder.add_meta_graph_and_variables(
#         sess,
#         self._input_saved_model_tags,
#         signature_def_map=self._grappler_meta_graph_def.signature_def)
# # Ignore other meta graphs from the input SavedModel.
# saved_model_builder.save()


# self._grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
# self._grappler_meta_graph_def.graph_def.CopyFrom(new_graph_def)
# self._grappler_meta_graph_def.signature_def[
#     self._input_saved_model_signature_key].CopyFrom(input_signature_def)

# self._converted_graph_def = new_graph_def