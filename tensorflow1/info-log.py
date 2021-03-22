import numpy as np

# IPU imports
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Configure argument for targeting the IPU
cfg = utils.create_ipu_config(profiling=True, 
                              # use_poplar_text_report=True,
                              enable_poplar_serialized_graph=True, 
                              profile_execution = True,
                              # enable_ipu_events = True,
                              report_directory = "./ipu-info")
cfg = utils.set_ipu_model_options(cfg, compile_ipu_code=False)
cfg = utils.auto_select_ipus(cfg, 1)
utils.configure_ipu_system(cfg)

with tf.device("cpu"):
  pa = tf.placeholder(np.float32, [2], name="a")
  pb = tf.placeholder(np.float32, [2], name="b")
  pc = tf.placeholder(np.float32, [2], name="c")
  # c = tf.placeholder(np.float32, [], name="c_out")

  # Create a trace event
  report = gen_ipu_ops.ipu_event_trace()


def basic_graph(pa, pb, pc):
  # Do basic addition with tensors
  o1 = pa + pb
  o2 = pa + pc
  simple_graph_output = o1 + o2
  simple_graph_output = tf.reduce_sum(simple_graph_output)
  return simple_graph_output


with ipu_scope("/device:IPU:0"):
  result = basic_graph(pa, pb, pc)

# tf.summary.scalar('c_out', result)
# ipu.summary_ops.ipu_compile_summary('report', [result])
# all_sum = tf.summary.merge_all()


with tf.Session() as sess:
  # f = tf.summary.FileWriter('logs', tf.get_default_graph())
  # Run the graph through the session feeding it an arbitrary dictionary
  result = sess.run(result,
                    feed_dict={
                        pa: [1., 1.],
                        pb: [0., 1.],
                        pc: [1., 5.],
                    })
  # sum_out, result = sess.run([all_sum, result],
  #                   feed_dict={
  #                       pa: [1., 1.],
  #                       pb: [0., 1.],
  #                       pc: [1., 5.],
  #                   })

  # Generate report based on the event run in session
  trace_out = sess.run(report)
  trace_report = utils.extract_all_strings_from_event_trace(trace_out)
  
  # rep = sess.run(report)
  compile_reports = ipu.utils.extract_compile_reports(trace_out)
  execute_reports = ipu.utils.extract_execute_reports(trace_out)
  poplar_graphs = ipu.utils.extract_poplar_serialized_graphs(trace_out)
  events = ipu.utils.extract_all_events(trace_out)

  print(compile_reports)
  print(execute_reports)
  print(poplar_graphs)
  print(events)

  # sum_out = sess.run(all_sum, feed_dict={c: sum(result)})
  # f.add_summary(sum_out, 0)

  # print("c = {}".format(result))

  # Write trace report to file
  with open('Trace_Event_Report.rep', "w") as f:
    f.write(trace_report)

  # Print the result
  print(result)