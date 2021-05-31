import onnx
# import onnxruntime as ort
from onnx.helper import make_node, make_attribute, make_model, make_tensor, make_graph
from onnx import TensorProto, SparseTensorProto, AttributeProto, ValueInfoProto, \
    TensorShapeProto, NodeProto, ModelProto, GraphProto, OperatorSetIdProto, \
    TypeProto, SequenceProto, MapProto, IR_VERSION



inputs = make_tensor(name="TensorDict/StandardKvParser_7:0", 
                     data_type=TensorProto.INT64, dims=)





graph = make_graph()
model = make_model()