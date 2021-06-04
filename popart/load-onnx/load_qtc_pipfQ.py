import popart
import numpy as np
from typing import *
import time
import onnx
from itertools import chain

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

def run(builder, opts, batch_size = 1, batch_per_step = 1, n_sample = None):

    global_batch_size = batch_per_step * batch_size
    n_sample = n_sample or global_batch_size

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

    # opts = popart.SessionOptions()
    # opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    # opts.enablePipelining = True
    # partials_type = "half"
    # opts.partialsTypeMatMuls = partials_type
    # opts.convolutionOptions = {'partialsType': partials_type}
    # opts.groupHostSync = True

    # builder.virtualGraph("Reshape_1:0", 0)
    # builder.virtualGraph("Reshape_2:0", 1)  
    # builder.virtualGraph("Reshape:0", 2)
    # builder.virtualGraph("prob", 3)
    # builder.pipelineStage("Reshape_1:0", 0)
    # builder.pipelineStage("Reshape_2:0", 1)
    # builder.pipelineStage("Reshape:0", 2)
    # builder.pipelineStage("prob", 3)

    # session = popart.InferenceSession(builder.getModelProto(), dataflow, device, inputShapeInfo)
    session = popart.InferenceSession(builder.getModelProto(), dataflow, device, 
                                    inputShapeInfo=inputShapeInfo,
                                    userOptions=opts)

    session.prepareDevice()
    anchors = session.initAnchorArrays()

    durations = []

    for feed_dict in fake_dataset(inputs_tensor_id, inputs_shapes, inputs_dtypes, num_samples=n_sample):

        print("[qtcv-inference] Starting batch inference")
        stepio = popart.PyStepIO(feed_dict, anchors)
        # start = time.perf_counter()
        session.run(stepio)
        for k, v in anchors.items():
            print(v)
        # duration = time.perf_counter() - start

        # durations.append(duration / global_batch_size)

    # np_dur = np.array(durations[10:]).mean()
    # print(f"Latency: {np_dur} s/sample(mean)")

    # for k, v in anchors.items():
    #     print(v)



def main():

    batch_size = 8
    batch_per_step = 20
    global_batch_size = batch_per_step * batch_size
    epochs = 80
    n_sample = epochs + 10 * 2

    # batch_size = 1
    # batch_per_step = 2
    # global_batch_size = batch_per_step * batch_size
    # # epochs = 80
    # n_sample = 20

    # builder = popart.Builder("qtc35-onnx-pipeline/model-on-half.onnx")
    # builder = popart.Builder("qtc35-onnx-pipeline/model.onnx")
    # builder = popart.Builder("qtc35/model.onnx")
    # builder = popart.Builder("subqtc-manually.onnx")
    builder = popart.Builder("qtc211-tf-ipu-fp16/qtcv211-int32-manual.onnx")
    # builder = popart.Builder("qtc211-tf-ipu-fp16/qtcv211-int32-manual.onnx")
    # builder = popart.Builder("qtc211-tf-ipu-fp16/qtcv211-wo-kv.onnx")
    # builder = popart.Builder("reproducer/sub_qtcv211.onnx")
    anchors = make_an_anchor(builder)
    inputs_tensor_id, inputShapeInfo, inputs_shapes, inputs_dtypes = add_shapeinfo_from_onnx(builder, batch_size=batch_size, batch_per_step = batch_per_step)
    # anchors = {output_name: popart.AnchorReturnType("All") for output_name in builder.getOutputTensorIds() }
    dataflow = popart.DataFlow(batch_per_step, anchors)
    device = popart.DeviceManager().acquireAvailableDevice(2)
    # device = popart.DeviceManager().createCpuDevice()

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto
    # opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    # opts.enablePipelining = True
    partials_type = "half"
    opts.partialsTypeMatMuls = partials_type
    opts.convolutionOptions = {'partialsType': partials_type}
    # opts.groupHostSync = True

    # onnx_m = onnx.load_from_string(builder.getModelProto())

    # reload_qt_ns = [n.output for n in onnx_m.graph.node if n.name.startswith("reload_qt/")]
    # reload_ns = [n.output for n in onnx_m.graph.node if n.name.startswith("reload/")]
    # rest_ns = [n.output for n in onnx_m.graph.node if not n.name.startswith("reload")]

    # for Oqt in chain(*reload_qt_ns):
    #     builder.virtualGraph(Oqt, 0)

    # for Ore in chain(*reload_ns):
    #     try:
    #         builder.virtualGraph(Oqt, 1)
    #     except:
    #         pass

    # for Oqt in chain(*rest_ns[1:]):
    #     builder.virtualGraph(Oqt, 1)
    
    # builder.virtualGraph(rest_ns[0][0], 0)

    # builder.virtualGraph("Reshape_1:0", 0)
    # builder.virtualGraph("Reshape_2:0", 1)  
    # builder.virtualGraph("Reshape:0", 2)
    # builder.virtualGraph("prob", 3)
    # builder.pipelineStage("Reshape_1:0", 0)
    # builder.pipelineStage("Reshape_2:0", 1)
    # builder.pipelineStage("Reshape:0", 2)
    # builder.pipelineStage("prob", 3)

    # session = popart.InferenceSession(builder.getModelProto(), dataflow, device, inputShapeInfo)
    session = popart.InferenceSession(builder.getModelProto(), dataflow, device, 
                                    inputShapeInfo=inputShapeInfo,
                                    userOptions=opts)

    # feed_dict = { i: np.random.randint(12, size=s).astype(convert_numpy_dtype(d)) for i, s, d in zip(inputs_tensor_id, inputs_shapes, inputs_dtypes) }

    session.prepareDevice()
    anchors = session.initAnchorArrays()

    durations = []
    tputs = []

    for feed_dict in fake_dataset(inputs_tensor_id, inputs_shapes, inputs_dtypes, num_samples=n_sample):

        print("[qtcv35-inference] Starting batch inference")
        stepio = popart.PyStepIO(feed_dict, anchors)
        start = time.perf_counter()
        session.run(stepio)
        duration = time.perf_counter() - start
        for k, v in anchors.items():
            print(v)

        # durations.append(duration / global_batch_size)
        durations.append(duration / batch_per_step)
        tputs.append(global_batch_size / duration)

    np_dur = np.array(durations[10:]).mean()
    np_tput = np.array(tputs[10:]).mean()
    print(f"Latency: {np_dur} s / batch (mean)")
    print(f"Troughput: {np_tput} samples / sec (mean)")

    # for k, v in anchors.items():
    #     print(v)

if __name__ == '__main__':
    main()