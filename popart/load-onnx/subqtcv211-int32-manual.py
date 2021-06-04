#%%
import popart
import onnx
from onnx.helper import make_graph, make_attribute, make_node
from onnx import TensorProto, SequenceProto
from onnx.checker import check_model
from onnx import numpy_helper
import numpy as np
from typing import *
import time
import onnxruntime as rt

# %%
def find_node_by_name(model, name: str):
    return [(i, n) for i, n in enumerate(model.graph.node) if n.name == name]

def find_node_by_input(model, name: str):
    return [(i, n) for i, n in enumerate(model.graph.node) if name in n.input]

def find_node_by_output(model, name: str):
    return [(i, n) for i, n in enumerate(model.graph.node) if name in n.output]

def find_init_by_name(model, name: str):
    return [(i, n) for i, n in enumerate(model.graph.initializer) if name == n.name]

def assign_init(inits, idx, data):
    d = inits[idx]
    inits.insert(idx, data)
    inits.remove(d)

def cast_init_int32(m, name: str):
    k = find_init_by_name(m, name)
    if len(k) == 1:
        i, init = k[0]
        data = numpy_helper.to_array(init)
        int32_data = numpy_helper.from_array(data.astype('int32'), name=name)
        # m.graph.initializer[i] = int32_data
        assign_init(m.graph.initializer, i, int32_data)

# %%
m = onnx.load("reproducer/sub_qtcv211.onnx")
check_model(m, 1)

# %%
m.graph.input[0].type.tensor_type.elem_type = 6
# %%
find_node_by_name(m, "reload_query/NotEqual")
# %%
find_init_by_name(m, "reload_qt/NotEqual/y:0")
# %%
# m.graph.initializer[6].data_type = TensorProto.INT32
cast_init_int32(m, "reload_qt/NotEqual/y:0")
# %%
check_model(m, 1)
# %%
find_node_by_output(m, "reload_query/concat:0")

# %%
m.graph.node[9].input[0] = "reload_query/concat_int32:0"

#%%
cast_op = make_node("Cast", inputs=["reload_query/concat:0"], outputs=["reload_query/concat_int32:0"], to = 6)
m.graph.node.insert(9, cast_op)
# %%
find_node_by_name(m, "Cast__711")
# %%
m.graph.node[41].attribute[0].i = 6

#%%
find_node_by_name(m, "reload_query/strided_slice_1")

#%%
m.graph.node[8].input[0]="reload_query/strided_slice_1_int32:0"
#%%
cast_op = make_node("Cast", inputs=["reload_query/strided_slice_1:0"], outputs=["reload_query/strided_slice_1_int32:0"], to = 6)
m.graph.node.insert(8, cast_op)
# %%
find_node_by_name(m, "reload_query/ToInt64_1")
#%%
m.graph.node[22].attribute[0].i = 6
#%%
find_init_by_name(m, "reload_query/sub/x:0")
#%%
# m.graph.initializer[0].data_type = TensorProto.INT32
cast_init_int32(m, "reload_query/sub/x:0")
#%%
find_init_by_name(m, "reload_query/mul/y:0")
#%%
# m.graph.initializer[2].data_type = TensorProto.INT32
cast_init_int32(m, "reload_query/mul/y:0")
#%%
find_init_by_name(m, "reload_query/mul_2/y:0")
#%%
# m.graph.initializer[1].data_type = TensorProto.INT32
cast_init_int32(m, "reload_query/mul_2/y:0")
#%%

#%%
sess = rt.InferenceSession("subqtc-manually.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: np.random.randint(10, size = [1, 35]).astype(np.int32)})[0]
print(pred_onx)
# %%
onnx.save(m, "subqtc-manually.onnx")
# %%
mm = onnx.load("subqtc-manually.onnx")
# %%
find_node_by_name(mm, "reload_query/zeros_like")
# %%
mm.graph.node[44].attribute[0].i = 6
# %%
onnx.save(mm, "subqtc-manually.onnx")
# %%
