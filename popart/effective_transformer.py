import numpy as np
import popart
from itertools import accumulate

np.random.seed(1)

def reshape_to_matrix(input_tensor):
    # return [B*F, N*H]
    pass

def reshape_from_matrix(input_tensor):
    pass

def kv_subgraph(builder, N = 6, H = 2):
    subgraph_builder = builder.createSubgraphBuilder()

    # kv_weight_shape = popart.TensorInfo("FLOAT", [N*H, N*H*2])
    kv_weight_shape = np.random.randint(10, size=(N*H, N*H)).astype(np.float32)

    kv_weight = subgraph_builder.addInitializedInputTensor(kv_weight_shape, "kv_weight")

    to_tensor_ = subgraph_builder.addUntypedInputTensor()
    out = subgraph_builder.aiOnnx.matmul([to_tensor_, kv_weight])

    subgraph_builder.addOutputTensor(out)

    return subgraph_builder

def q_subgraph(builder, N = 6, H = 2):

    subgraph_builder = builder.createSubgraphBuilder()

    # q_weight_shape = popart.TensorInfo("FLOAT", [N*H, N*H])
    q_weight_shape = np.random.randint(10, size=(N*H, N*H)).astype(np.float32)

    q_weight = subgraph_builder.addInitializedInputTensor(q_weight_shape, "q_weight")

    to_tensor_ = subgraph_builder.addUntypedInputTensor()
    out = subgraph_builder.aiOnnx.matmul([to_tensor_, q_weight])

    subgraph_builder.addOutputTensor(out)

    return subgraph_builder

def _gen_packed_sequences(lengths=[1,2,3,4,5,2], shape=[10,], dtype=np.float32):
    sequences = []
    for length in lengths:
        sequence_shape = [length] + shape
        sequence = np.random.rand(*sequence_shape).astype(np.float32)
        sequences.append(sequence)
    offsets = [0] + list(accumulate(lengths[:-1]))
    return np.concatenate(sequences), offsets

def _unpack(data, offsets, sequenceLength):
    total_row = data.shape[0]
    offsets_all = offsets + [total_row]
    lengths = [*map(lambda x : x[1] - x[0], zip(offsets_all, offsets_all[1:]))]
    sequences = []
    for offset, length in zip(offsets, lengths):
        sequence = data[offset:offset + length]
        padding = [[0, 0] for _ in range(sequence.ndim)]
        padding[0][1] = sequenceLength - sequence.shape[0]
        sequence = np.pad(sequence, padding)
        sequences.append(sequence)
    return np.stack(sequences)


def main():
    builder = popart.Builder()

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    
    C = 17  # after compressed
    B = 72
    F = 12
    T = 12
    N = 5
    H = 2

    maxSequenceLength = 12

    from_tensor_shape = popart.TensorInfo("FLOAT", [C, N*H])
    to_tensor_shape = popart.TensorInfo("FLOAT", [C, N*H])

    from_tensor = builder.addInputTensor(from_tensor_shape, "from_tensor")
    to_tensor = builder.addInputTensor(to_tensor_shape, "to_tensor")

    from_tensor_sequenceLengths_shape = popart.TensorInfo("UINT32", [B])
    from_tensor_sequenceOffsets_shape = popart.TensorInfo("UINT32", [B])
    to_tensor_sequenceLengths_shape = popart.TensorInfo("UINT32", [B])
    to_tensor_sequenceOffsets_shape = popart.TensorInfo("UINT32", [B])

    from_tensor_sequenceLengths = builder.addInputTensor(from_tensor_sequenceLengths_shape, "from_tensor_sequenceLengths")
    from_tensor_sequenceOffsets = builder.addInputTensor(from_tensor_sequenceOffsets_shape, "from_tensor_sequenceOffsets")

    to_tensor_sequenceLengths = builder.addInputTensor(to_tensor_sequenceLengths_shape, "to_tensor_sequenceLengths")
    to_tensor_sequenceOffsets = builder.addInputTensor(to_tensor_sequenceOffsets_shape, "to_tensor_sequenceOffsets")


    query_layer_graph = q_subgraph(builder)
    kv_layer_graph = kv_subgraph(builder)

    q_out = builder.aiGraphcore.packedDataBlock([
        from_tensor, from_tensor_sequenceOffsets, from_tensor_sequenceLengths, from_tensor_sequenceOffsets,
        from_tensor_sequenceLengths
    ], [maxSequenceLength], C, 1, query_layer_graph)

    kv_out = builder.aiGraphcore.packedDataBlock([
        to_tensor, to_tensor_sequenceOffsets, to_tensor_sequenceLengths, to_tensor_sequenceOffsets,
        to_tensor_sequenceLengths
    ], [maxSequenceLength], C, 1, kv_layer_graph)

    builder.addOutputTensor(q_out)
    builder.addOutputTensor(kv_out)

    # Get the ONNX protobuf from the builder to pass to the Session
    proto = builder.getModelProto()

    # Create a runtime environment
    anchors = {q_out: popart.AnchorReturnType("All"), kv_out:popart.AnchorReturnType("All")}
    dataFlow = popart.DataFlow(1, anchors)
    device = popart.DeviceManager().acquireAvailableDevice(request_ipus=1)

    # Create the session from the graph, data feed and device information
    session = popart.InferenceSession(proto, dataFlow, device)

    session.prepareDevice()

    d, o = _gen_packed_sequences()
    o_np = np.array(o, dtype=np.uint32)
    lens = np.array([1,2,3,4,5,2], dtype=np.uint32)

    anchors = session.initAnchorArrays()

    feed_dict={
        from_tensor: d,
        from_tensor_sequenceOffsets: o_np,
        from_tensor_sequenceLengths: lens,
        to_tensor: d,
        to_tensor_sequenceOffsets: o_np,
        to_tensor_sequenceLengths: lens,
    }

    stepio = popart.PyStepIO(feed_dict, anchors)

    session.run(stepio)

    print(anchors[o])
    print()
    print(anchors[o])

if __name__ == '__main__':
    main()