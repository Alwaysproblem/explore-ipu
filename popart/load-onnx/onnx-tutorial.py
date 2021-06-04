#%%
import onnx
# import onnxruntime as ort
from onnx.helper import make_node, make_attribute, make_model, make_tensor, make_graph, make_tensor_value_info, make_operatorsetid
from onnx import TensorProto, SparseTensorProto, AttributeProto, ValueInfoProto, \
    TensorShapeProto, NodeProto, ModelProto, GraphProto, OperatorSetIdProto, \
    TypeProto, SequenceProto, MapProto, IR_VERSION
import numpy as np
from itertools import chain
import onnxruntime as rt
#%%
conv = make_node("Conv", inputs=["x:0", "w:0", "b:0"], outputs=["y:0"], name = "conv",
                 dilations=[1,1], group=1, kernel_shape=[3, 3], pads=[1,1,1,1], strides=[1,1])
print(conv)
#%%
relu = make_node("Relu", inputs=["y:0"], outputs=["relu:0"], name = "relu")
print(relu)
#%%

pool = make_node("MaxPool", inputs=["relu:0"], outputs=["pool:0"], name="pooling",
                  kernel_shape = [2, 2], strides = [2, 2])
print(pool)

#%%
w = np.load("reproducer/reload_qt_semantics_cnn_conv_kernel_0_32_weights_read__432.npy").astype(np.float32)
b = np.array([
    -0.03277587890625,
    -0.0138092041015625,
    -0.07452392578125,
    0.02117919921875,
    -0.043212890625,
    -0.0026798248291015625,
    0.01161956787109375,
    -0.00545501708984375,
    -0.022216796875,
    -0.043060302734375,
    -0.037567138671875,
    0.0239105224609375,
    0.01352691650390625,
    0.00922393798828125,
    -0.04510498046875,
    -0.0019168853759765625,
    -0.0367431640625,
    -0.0211944580078125,
    -0.006763458251953125,
    0.0163726806640625,
    -0.040924072265625,
    -0.026031494140625,
    -0.0523681640625,
    -0.050201416015625,
    -0.05413818359375,
    -0.0281829833984375,
    0.0109710693359375,
    -0.0199432373046875,
    -0.0001138448715209961,
    0.009918212890625,
    -0.032867431640625,
    -0.0261077880859375
], dtype = np.float32)

#%%
initializers = [
    make_tensor(name="w:0", data_type=TensorProto.FLOAT, dims=w.shape, vals=w.flatten().tolist()),
    make_tensor(name="b:0", data_type=TensorProto.FLOAT, dims=b.shape, vals=b.flatten().tolist())
]

#%%
inputs = [make_tensor_value_info(name="x:0", elem_type=TensorProto.FLOAT, shape=[1, 1, 35, 61])]
outputs = [make_tensor_value_info(name="pool:0", elem_type=TensorProto.FLOAT, shape=[1, 32, 17, 30])]

graph = make_graph([conv, relu, pool], name="reproducer", inputs=inputs, outputs=outputs, initializer=initializers)

#%%
versions = make_operatorsetid(domain="", version=11)
model = make_model(graph, ir_version=6, opset_imports = [versions])
# %%
onnx.checker.check_model(model, 1)
# %%
sess = rt.InferenceSession(model.SerializeToString())
# %%
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
# %%
pred_onx = sess.run(
    [label_name], {input_name: np.random.random(size=[1, 1, 35, 61]).astype(np.float32)})[0]
print(pred_onx)
# %%
