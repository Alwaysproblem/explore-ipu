#%%
import onnx
import onnxruntime as rt
from onnx.helper import (make_node, make_attribute, make_model, make_tensor, 
                        make_graph, make_tensor_value_info, make_operatorsetid)
from onnx import TensorProto, SparseTensorProto, AttributeProto, ValueInfoProto, \
    TensorShapeProto, NodeProto, ModelProto, GraphProto, OperatorSetIdProto, \
    TypeProto, SequenceProto, MapProto, IR_VERSION
import numpy as np
from copy import deepcopy

#%%
m = onnx.load("qtc211-tf-ipu/qtcv211-wo-kv.onnx")
#%%
select_node_name = set([
        "reload_query/bert/embeddings/add",
        "reload_query/bert/embeddings/Reshape_2",
        "reload_query/bert/embeddings/MatMul",
        "reload_query/bert/embeddings/Reshape_2__767",
        "reload_query/bert/embeddings/one_hot",
        "reload_query/bert/embeddings/Reshape_2/shape_Concat__766",
        "reload_query/bert/embeddings/strided_slice_1",
        "reload_query/bert/embeddings/Reshape_1",
        "reload_query/bert/embeddings/Shape_1__758",
        "reload_query/zeros_like",
        "reload_query/bert/embeddings/Shape_1",
        "Mul__714",
        "reload_query/bert/embeddings/Reshape",
        "Cast__711",
        "reload_query/bert/embeddings/embedding_lookup",
        "reload_query/bert/embeddings/Reshape__757",
        "reload_query/bert/embeddings/Reshape/shape_Concat__756",
        "reload_query/bert/embeddings/strided_slice",
        "reload_query/bert/embeddings/Shape",
        "reload_query/bert/embeddings/Shape__748",
        "reload_query/bert/embeddings/ExpandDims",
        "reload_query/add_1",
        "Concat__1254",
        "reload_query/mul_3",
        "reload_query/mul_2",
        "reload_query/sub",
        "reload_query/mul_1",
        "reload_query/ToInt64_1",
        "reload_query/LogicalAnd",
        "reload_query/Equal",
        "reload_query/SequenceMask_1/Less",
        "reload_query/concat",
        "reload_query/SequenceMask_1/Less__732",
        "reload_query/strided_slice_2",
        "reload_query/strided_slice_1",
        "reload_query/SequenceMask/Cast",
        "reload_query/mul",
        "reload_query/SequenceMask/ExpandDims",
        "reload_query/ones_like",
        "reload_query/add",
        "reload_query/ones_like__725",
        "reload_query/Sum",
        "reload_query/ones_like/Shape__724",
        "reload_query/Cast",
        "reload_query/ones_like/Shape",
        "reload_query/NotEqual__729",
        "reload_query/NotEqual",
        "reload_query/strided_slice",
    ])

#%%
inputs = deepcopy(m.graph.input[2])
# %%
graph = make_graph(nodes=[], name="reproducer", inputs=[inputs], outputs = [])

#%%
for n in m.graph.node:
    if n.name in select_node_name:
        graph.node.append(deepcopy(n))

#%%
inits = []
inits_names = set([init.name for init in m.graph.initializer])
for n in graph.node:
    for inp in n.input:
        if inp in inits_names:
            inits.append(inp)
# %%
for init in m.graph.initializer:
    if init.name in set(inits):
        graph.initializer.append(deepcopy(init))
# %%
outputs = make_tensor_value_info(name="reload_query/bert/embeddings/add:0", 
                elem_type=TensorProto.FLOAT16, shape = [1, 35, 768])
# %%
graph.output.append(outputs)
# %%
opset = make_operatorsetid(domain="", version=11)
#%%
model = make_model(graph, ir_version=6, opset_imports = [opset])
# %%
onnx.checker.check_model(model)
onnx.save(model, "sub_qtcv211.onnx")
# %%
print(onnx.helper.printable_graph(model.graph))
#%%
sess = rt.InferenceSession("sub_qtcv211.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: np.random.randint(10, size = [1, 35]).astype(np.int64)})[0]
print(pred_onx)