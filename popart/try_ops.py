import popart
import numpy as np



source_shape, dest_shape, N, source_offset, dest_offset = [6, 6], [1, 6], [1], [0], [0]

source = np.arange(np.prod(source_shape)) + 10
source = np.reshape(source, source_shape).astype(np.float32)

dest = np.zeros(dest_shape, np.float32)
N = np.asarray(N, np.uint32)
source_offset = np.asarray(source_offset, np.uint32)
dest_offset = np.asarray(dest_offset, np.uint32)

builder = popart.Builder(opsets={
    "ai.onnx": 9,
    "ai.onnx.ml": 1,
    "ai.graphcore": 1
})

source_id = builder.addInputTensor(popart.TensorInfo("FLOAT", source_shape), debugContext="source_id")
dest_id = builder.addInputTensor(popart.TensorInfo("FLOAT", dest_shape), debugContext="dest_id")
N_id = builder.addInputTensor(popart.TensorInfo("UINT32", [len(N)]), debugContext="N_id")

source_offset_id = builder.addInputTensor(popart.TensorInfo("UINT32", [len(source_offset)]), debugContext="source_offset_id")
dest_offset_id = builder.addInputTensor(popart.TensorInfo("UINT32", [len(dest_offset)]), debugContext="dest_offset_id")

o = builder.aiGraphcore.sequenceslice(
    [source_id, dest_id, N_id, source_offset_id, dest_offset_id],
    True,
    debugContext="sequenceslice")

builder.addOutputTensor(o)


proto = builder.getModelProto()

anchors = {o: popart.AnchorReturnType("All")}
dataFlow = popart.DataFlow(1, anchors)
device = popart.DeviceManager().acquireAvailableDevice(1)

session = popart.InferenceSession(proto, dataFlow, device)

session.prepareDevice()

anchors = session.initAnchorArrays()

inputs = {
    "source_id": source,
    "dest_id": dest,
    "N_id": N,
    "source_offset_id": source_offset,
    "dest_offset_id": dest_offset,
}

stepio = popart.PyStepIO(inputs, anchors)

session.run(stepio)

print(source)

print(anchors[o])