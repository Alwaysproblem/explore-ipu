import popart
import onnx 
from onnx.tools import update_model_dims
import numpy as np
from typing import *


def make_an_anchor(onnx_model_builder: popart.Builder, anchor_return_type: str = "ALL"):
    return {output_name: popart.AnchorReturnType(anchor_return_type) 
                for output_name in onnx_model_builder.getOutputTensorIds()}

def set_batch_size(shape: List, batch_size = 1):
    """ shape is like [unknow_batch_size, n1, n2, ...] """
    shape[0] = batch_size if shape[0] == 0 else shape[0]
    return shape

def convert_popart_dtype(dtype: str):
    dtype_conv_dic = {
        "int64": "INT32",
        "int32": "INT32",
        "float32": "FLOAT",
        "float16": "FLOAT16",
        "float64": "FLOAT",
    }

    return dtype_conv_dic[dtype]

def convert_numpy_dtype(dtype: str):
    dtype_conv_dic = {
        "int64": np.int32, 
        "int32": np.int32, 
        "float32": np.float32,
        "float": np.float32,
        "float64": np.float32,
    }
    return dtype_conv_dic[dtype.lower()]


def add_shapeinfo_from_onnx(onnx_model_builder: popart.Builder, batch_size = 1):
    inputs_tensor_id = onnx_model_builder.getInputTensorIds()
    outputs_tensor_id = onnx_model_builder.getOutputTensorIds()

    inputs_shapes = [set_batch_size(onnx_model_builder.getTensorShape(i), 
                                batch_size=batch_size) for i in inputs_tensor_id]
    inputs_dtypes = [convert_popart_dtype(onnx_model_builder.getTensorDtypeString(i)) for i in inputs_tensor_id]

    inputs_tensor_shapes = [set_batch_size(onnx_model_builder.getTensorShape(i), 
                                batch_size=batch_size) for i in inputs_tensor_id + outputs_tensor_id]
    inputs_tensor_dtypes = [convert_popart_dtype(onnx_model_builder.getTensorDtypeString(i)) for i in inputs_tensor_id + outputs_tensor_id]

    inputShapeInfo = popart.InputShapeInfo()

    for tid, tshape, tdype in zip(inputs_tensor_id + outputs_tensor_id, inputs_tensor_shapes, inputs_tensor_dtypes):
        inputShapeInfo.add(tid, popart.TensorInfo(tdype, tshape))

    return inputs_tensor_id, inputShapeInfo, inputs_shapes, inputs_dtypes


# bert_model = onnx.load("bertsquad-10/bertsquad10.onnx")

# input_shape = {
#     "unique_ids_raw_output___9:0": [3, ],
#     "segment_ids:0": [3, 256],
#     "input_mask:0": [3, 256],
#     "input_ids:0": [3, 256],
#     }
# output_shape = {
#     "unstack:1": [3, 256],
#     "unstack:0": [3, 256],
#     "unique_ids:0": [3],
    
# }

# bert = update_model_dims.update_inputs_outputs_dims(bert_model, input_shape, output_shape)

batch_size = 5

# transformed_mode = popart.GraphTransformer("bertsquad-10/bertsquad10.onnx")
# transformed_mode.convertINT64ToINT32()

builder = popart.Builder("qtc35/model.onnx")
anchors = make_an_anchor(builder)
inputs_tensor_id, inputShapeInfo, inputs_shapes, inputs_dtypes = add_shapeinfo_from_onnx(builder, batch_size)
# anchors = {output_name: popart.AnchorReturnType("All") for output_name in builder.getOutputTensorIds() }
dataflow = popart.DataFlow(1, anchors)
device = popart.DeviceManager().acquireAvailableDevice(2)
# device = popart.DeviceManager().createCpuDevice()

opts = popart.SessionOptions()
opts.virtualGraphMode = popart.VirtualGraphMode.Auto


session = popart.InferenceSession(builder.getModelProto(), dataflow, device, 
                                  inputShapeInfo=inputShapeInfo,
                                  userOptions=opts)

print(session)

feed_dict = { i: np.random.randint(12, size=s).astype(convert_numpy_dtype(d)) for i, s, d in zip(inputs_tensor_id, inputs_shapes, inputs_dtypes) }

# feed_dict={
#     "unique_ids_raw_output___9:0": np.random.randint(12, size=[batch_size,]).astype(np.int32),
#     "segment_ids:0": np.random.randint(12, size=[batch_size, 256]).astype(np.int32),
#     "input_mask:0": np.random.randint(12, size=[batch_size, 256]).astype(np.int32),
#     "input_ids:0": np.random.randint(12, size=[batch_size, 256]).astype(np.int32),
# }

session.prepareDevice()
anchors = session.initAnchorArrays()

stepio = popart.PyStepIO(feed_dict, anchors)

session.run(stepio)

for k, v in anchors.items():
    print(v)
