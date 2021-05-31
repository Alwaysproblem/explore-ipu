# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

# Create a builder and construct a graph
builder = popart.Builder()

data_shape = popart.TensorInfo("FLOAT", [1])

i1 = builder.addInputTensor(data_shape)
i2 = builder.addInputTensor(data_shape)

o = builder.aiOnnx.add([i1, i2])

builder.addOutputTensor(o)

proto = builder.getModelProto()

# Describe how to run the model
dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

# Create a session to compile and execute the graph
session = popart.InferenceSession(
    fnModel=proto,
    dataFlow=dataFlow,
    deviceInfo=popart.DeviceManager().createIpuModelDevice({}))

# Compile graph
session.prepareDevice()

# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()

# Generate some random input data
data_a = np.random.rand(1).astype(np.float32)
data_b = np.random.rand(1).astype(np.float32)

stepio = popart.PyStepIO({i1: data_a, i2: data_b}, anchors)
session.run(stepio)

print("Input a is " + str(data_a))
print("Input b is " + str(data_b))
print("Result is " + str(anchors[o]))