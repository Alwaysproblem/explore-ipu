# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
# import torch
# import torch.nn as nn
from itertools import accumulate

import popart

def _gen_packed_sequences(lengths, shape, dtype=np.float32):
    sequences = []
    for length in lengths:
        sequence_shape = [length] + shape
        sequence = np.random.rand(*sequence_shape).astype(np.float32)
        sequences.append(sequence)
    offsets = [0] + list(accumulate(lengths[:-1]))
    return np.concatenate(sequences), offsets


def _unpack(data, offsets, lengths, sequenceLength):
    sequences = []
    for offset, length in zip(offsets, lengths):
        sequence = data[offset:offset + length]
        padding = [[0, 0] for _ in range(sequence.ndim)]
        padding[0][1] = sequenceLength - sequence.shape[0]
        sequence = np.pad(sequence, padding)
        sequences.append(sequence)
    return np.stack(sequences)


def _pack(data, result, offsets, lengths):
    sequences = []
    for i in range(data.shape[0]):
        offset = offsets[i]
        length = lengths[i]
        sequence = data[i, :length]
        result[offset:offset + length] = sequence


# `inputs`: [data1, offsets1, lengths1, ..., dataN, offsetsN, lengthsN, resultOffsets, resultLengths].
# `maxSequenceLengths`: [int] of size number_of_data_inputs.
# `destination`: The array used for the result.
# `nSequencesPerInnerLoop`: Number of sequences to pass to the func.
# `func`: The function to run.
def _packed_data_block_reference(inputs, maxSequenceLengths, result_size,
                                 nSequencesPerInnerLoop, func):
    # get the result offsets and lengths
    assert len(inputs) > 2
    resultOffsets = inputs[-2]
    resultLengths = inputs[-1]
    inputs = inputs[:-2]

    # unpack each data input
    data_inputs = []
    assert len(inputs) % 3 == 0
    input_count = len(inputs) // 3
    for i in range(input_count):
        data = inputs[i * 3]
        offsets = inputs[(i * 3) + 1]
        lengths = inputs[(i * 3) + 2]
        maxSequenceLength = maxSequenceLengths[i]

        d = _unpack(data, offsets, lengths, maxSequenceLength)
        data_inputs.append(d)

    nSequences = len(data_inputs[0])

    results = []
    for i in range(nSequences // nSequencesPerInnerLoop):
        ins = []
        for di in data_inputs:
            ins.append(
                di[(i * nSequencesPerInnerLoop):(i * nSequencesPerInnerLoop) +
                   nSequencesPerInnerLoop])
        r = func(*ins)
        results.append(r)

    destination_shape = [result_size] + list(results[0].shape)[2:]
    destination = np.zeros(destination_shape).astype(results[0].dtype)

    for i in range(len(results)):
        for innerSequence in range(nSequencesPerInnerLoop):
            idx = i * nSequencesPerInnerLoop + innerSequence

            offset = resultOffsets[idx]
            length = resultLengths[idx]
            destination[offset:offset +
                        length] = results[i][innerSequence][:length]

    return destination


def test_packed_data_block_reference(nSequencesPerInnerLoop):
    sequenceLengths = [3, 5, 7, 4, 6, 2]
    data, sequenceOffsets = _gen_packed_sequences(sequenceLengths, [5])

    maxSequenceLength = 10

    unpacked_data = _unpack(data, sequenceOffsets, sequenceLengths,
                            maxSequenceLength)

    def unpacked_ref(data):
        dt = np.transpose(data, [0, 2, 1])
        mm = np.matmul(data, dt)
        return mm

    def packed_ref(data):
        calls_to_func = 0

        def func(d):
            nonlocal calls_to_func
            calls_to_func += 1

            dt = np.transpose(d, [0, 2, 1])
            return np.matmul(d, dt)

        result = _packed_data_block_reference([
            data, sequenceOffsets, sequenceLengths, sequenceOffsets,
            sequenceLengths
        ], [maxSequenceLength], data.shape[0], nSequencesPerInnerLoop, func)

        # Check how many times `func` was called.
        nSequences = len(sequenceLengths)
        assert calls_to_func == nSequences // nSequencesPerInnerLoop

        return result

    d = _unpack(data, sequenceOffsets, sequenceLengths, maxSequenceLength)
    unpacked_result = unpacked_ref(d)

    packed_result = packed_ref(data)
    packed_result = _unpack(packed_result, sequenceOffsets, sequenceLengths,
                            maxSequenceLength)

    assert unpacked_result.shape == packed_result.shape
    assert np.array_equal(packed_result, unpacked_result)


def test_packeddatablockop(op_tester):
    np.random.seed(0)

    sequenceLengths = [3, 5, 7, 4, 6, 2]
    data, sequenceOffsets = _gen_packed_sequences(sequenceLengths, [5])
    data = (data * 9 + 1).astype(np.uint32).astype(np.float32)

    sequenceLengths = np.array(sequenceLengths).astype(np.uint32)
    sequenceOffsets = np.array(sequenceOffsets).astype(np.uint32)

    maxSequenceLength = 10

    def init_builder(builder):
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
        return [out]

    def reference(ref_data):
        d = _unpack(data, sequenceOffsets, sequenceLengths, maxSequenceLength)
        dt = np.transpose(d, [0, 2, 1])
        mm = np.matmul(d, dt)
        result = np.zeros([27, 10]).astype(np.float32)
        _pack(mm, result, sequenceOffsets, sequenceLengths)
        return [result]

    op_tester.patterns.enablePattern("PackedDataBlock", True)
    op_tester.run(init_builder, reference, 'infer')


test_packed_data_block_reference(1)