import popart
import numpy as np


builder = popart.Builder(opsets={
    "ai.onnx": 9,
    "ai.onnx.ml": 1,
    "ai.graphcore": 1
})

dataId = builder.addInputTensor(data, "data")
sequenceLengthsId = builder.addInputTensor(sequenceLengths, "lengths")
sequenceOffsetsId = builder.addInputTensor(sequenceOffsets, "offsets")

subgraph_builder = builder.createSubgraphBuilder()

sgi0 = subgraph_builder.addUntypedInputTensor()

dt = subgraph_builder.aiOnnx.transpose([sgi0], [0, 2, 1])
out = subgraph_builder.aiOnnx.matmul([sgi0, dt])

subgraph_builder.addOutputTensor(out)

out = builder.aiGraphcore.packedDataBlock([
    dataId, sequenceOffsetsId, sequenceLengthsId, sequenceOffsetsId,
    sequenceLengthsId
], [maxSequenceLength], data.shape[0], 1, subgraph_builder)

builder.addOutputTensor(out)



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