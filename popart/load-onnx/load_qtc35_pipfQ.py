import popart
import numpy as np
from typing import *
import time

np.random.seed(1)

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


def add_shapeinfo_from_onnx(onnx_model_builder: popart.Builder, batch_size = 1, batch_per_step = 1):
    inputs_tensor_id = onnx_model_builder.getInputTensorIds()
    outputs_tensor_id = onnx_model_builder.getOutputTensorIds()

    print(inputs_tensor_id)
    inputs_shapes = [set_batch_size(onnx_model_builder.getTensorShape(i), 
                                batch_size=batch_size * batch_per_step) for i in inputs_tensor_id]
    print(inputs_shapes)
    inputs_dtypes = [convert_popart_dtype(onnx_model_builder.getTensorDtypeString(i)) for i in inputs_tensor_id]
    print(inputs_dtypes)

    inputs_tensor_shapes = [set_batch_size(onnx_model_builder.getTensorShape(i), 
                                batch_size=batch_size) for i in inputs_tensor_id + outputs_tensor_id]
    inputs_tensor_dtypes = [convert_popart_dtype(onnx_model_builder.getTensorDtypeString(i)) for i in inputs_tensor_id + outputs_tensor_id]

    inputShapeInfo = popart.InputShapeInfo()

    for tid, tshape, tdype in zip(inputs_tensor_id + outputs_tensor_id, inputs_tensor_shapes, inputs_tensor_dtypes):
        inputShapeInfo.add(tid, popart.TensorInfo(tdype, tshape))

    return inputs_tensor_id, inputShapeInfo, inputs_shapes, inputs_dtypes


def fake_dataset(inputs_tensor_id, inputs_shapes, inputs_dtypes, num_samples = 100):
    for _ in range(num_samples):
        yield { i: np.random.randint(12, size=s).astype(convert_numpy_dtype(d)) for i, s, d in zip(inputs_tensor_id, inputs_shapes, inputs_dtypes) }

def main():

    # batch_size = 5
    # batch_per_step = 20
    # global_batch_size = batch_per_step * batch_size
    # epochs = 80
    # n_sample = epochs + 10 * 2

    batch_size = 1
    batch_per_step = 1
    global_batch_size = batch_per_step * batch_size
    # epochs = 80
    n_sample = 1

    builder = popart.Builder("qtc35/model.onnx")
    # builder = popart.Builder("subqtc-manually.onnx")
    # builder = popart.Builder("qtc211-tf-ipu/qtcv211-int32-manual.onnx")
    # builder = popart.Builder("reproducer/sub_qtcv211.onnx")
    anchors = make_an_anchor(builder)
    inputs_tensor_id, inputShapeInfo, inputs_shapes, inputs_dtypes = add_shapeinfo_from_onnx(builder, batch_size=batch_size, batch_per_step = batch_per_step)
    # anchors = {output_name: popart.AnchorReturnType("All") for output_name in builder.getOutputTensorIds() }
    dataflow = popart.DataFlow(batch_per_step, anchors)
    device = popart.DeviceManager().acquireAvailableDevice(2)
    # device = popart.DeviceManager().createCpuDevice()

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto
    # opts.enablePipelining = True
    partials_type = "half"
    opts.partialsTypeMatMuls = partials_type
    opts.convolutionOptions = {'partialsType': partials_type}
    # opts.groupHostSync = True

    # session = popart.InferenceSession(builder.getModelProto(), dataflow, device, inputShapeInfo)
    session = popart.InferenceSession(builder.getModelProto(), dataflow, device, 
                                    inputShapeInfo=inputShapeInfo,
                                    userOptions=opts)

    # feed_dict = { i: np.random.randint(12, size=s).astype(convert_numpy_dtype(d)) for i, s, d in zip(inputs_tensor_id, inputs_shapes, inputs_dtypes) }

    session.prepareDevice()
    anchors = session.initAnchorArrays()

    durations = []

    for feed_dict in fake_dataset(inputs_tensor_id, inputs_shapes, inputs_dtypes, num_samples=n_sample):

        print("[qtcv35-inference] Starting batch inference")
        stepio = popart.PyStepIO(feed_dict, anchors)
        start = time.perf_counter()
        session.run(stepio)
        for k, v in anchors.items():
            print(v)
        duration = time.perf_counter() - start

        durations.append(duration / global_batch_size)

    np_dur = np.array(durations[10:]).mean()
    print(f"Latency: {np_dur} s/sample(mean)")

    # for k, v in anchors.items():
    #     print(v)

if __name__ == '__main__':
    main()