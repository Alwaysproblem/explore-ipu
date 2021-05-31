import numpy as np
import onnxruntime as ort
import tokenization
import os
from run_onnx_squad import *
import json

input_file = 'inputs.json'
with open(input_file) as json_file:  
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))
  
# preprocess input
predict_file = 'inputs.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30


vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

my_list = []


# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input 
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer, 
                                                                              max_seq_length, doc_stride, max_query_length)

print(input_ids)