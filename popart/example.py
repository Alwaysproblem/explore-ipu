# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
builder = popart.Builder()

# Build a simple graph
i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

o = builder.aiOnnx.add([i1, i2])

builder.addOutputTensor(o)

# Get the ONNX protobuf from the builder to pass to the Session
proto = builder.getModelProto()

# Create a runtime environment
anchors = {o: popart.AnchorReturnType("All")}
dataFlow = popart.DataFlow(1, anchors)
device = popart.DeviceManager().createIpuModelDevice({})

# Create the session from the graph, data feed and device information
session = popart.InferenceSession(proto, dataFlow, device)

session.prepareDevice()

a = np.random.standard_normal((1,)).astype(np.float32)
b = np.random.standard_normal((1,)).astype(np.float32)

anchors = session.initAnchorArrays()

stepio = popart.PyStepIO({i1: a, i2: b}, anchors)

session.run(stepio)

print(anchors[o])


