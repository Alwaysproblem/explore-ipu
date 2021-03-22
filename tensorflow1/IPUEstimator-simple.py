#TODO: need to correct unconverge

import tensorflow.compat.v1 as tf
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as Split
import numpy as np
import time

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python import ipu

# K.set_floatx('float32')

# NUM_CLASSES = 2
NUM_REPLICAS = 1
# iterations_per_loop

def datasets_from_numpy(sample_n = 100):

  # generate some data 2-dimension. shape = (10, 2)
  meana = np.array([1, 1])
  cova = np.array([[0.5, 0],[0, 0.5]])

  meanb = np.array([2, 2])
  covb = np.array([[0.5, 0],[0, 0.5]])

  x_red = np.random.multivariate_normal(mean=meana, cov = cova, size=sample_n)
  x_green = np.random.multivariate_normal(mean=meanb, cov = covb, size=sample_n)

  y_red = np.array([1] * sample_n)
  y_green = np.array([0] * sample_n)

  # plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
  # plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
  # plt.show()

  X = np.concatenate([x_red, x_green])
  # X = np.concatenate([np.ones((sample_n*2, 1)), X], axis = 1)
  y = np.concatenate([y_red, y_green])
  y = np.expand_dims(y, axis = 1)

  X = X.astype(np.float32)
  y = y.astype(np.float32)

  X_train, X_test, y_train, y_test = Split(X, y, test_size=0.33, random_state=42)
  return X_train, X_test, y_train, y_test

def make_to_tensor(X_train, X_test, y_train, y_test):
  train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(X_train, dtype=tf.dtypes.float32), tf.constant(y_train, dtype=tf.dtypes.float32)))
  test_ds = tf.data.Dataset.from_tensor_slices((tf.constant(X_test, dtype=tf.dtypes.float32), tf.constant(y_test, dtype=tf.dtypes.float32)))

  train_ds = train_ds.shuffle(10)
  test_ds = test_ds.shuffle(10)

  train_ds = train_ds.repeat()
  test_ds = test_ds.repeat()

  train_ds = train_ds.batch(10, drop_remainder=True)
  test_ds = test_ds.batch(10, drop_remainder=True)

  return train_ds, test_ds


def model_fn(features, labels, mode, params,):
  model = Sequential()
  model.add(Dense(5))
  model.add(Activation("relu"))
  model.add(Dense(1))

  logits = model(features, training = (mode == tf.estimator.ModeKeys.TRAIN))

  loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

  if mode == tf.estimator.ModeKeys.EVAL:
      predictions = tf.round(logits)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(labels=labels,
                                          predictions=predictions),
      }
      return tf.estimator.EstimatorSpec(mode,
                                        loss=loss,
                                        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(params['lr'])
    if NUM_REPLICAS > 1:
      optimizer = ipu.cross_replica_optimizer.CrossReplicaOptimizer(optimizer)
    train_op = optimizer.minimize(loss=loss)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def create_ipu_estimator():
  ipu_options = ipu.utils.create_ipu_config(
      profiling=False,
      use_poplar_text_report=False,
  )

  ipu.utils.auto_select_ipus(ipu_options, num_ipus=NUM_REPLICAS)

  ipu_run_config = ipu.ipu_run_config.IPURunConfig(
      iterations_per_loop=100,
      num_replicas=NUM_REPLICAS,
      ipu_options=ipu_options,
  )

  config = ipu.ipu_run_config.RunConfig(
      ipu_run_config=ipu_run_config,
      log_step_count_steps=10,
      save_summary_steps=1,
      # model_dir="./models-tf1",
  )
  return ipu.ipu_estimator.IPUEstimator(
      config=config,
      model_fn=model_fn,
      params={
          "lr": 0.001,
      },
  )


def train(ipu_estimator):
  """Train a model on IPU and save checkpoints to the given `args.model_dir`."""
  def input_fn():
    # If using Dataset.from_tensor_slices(), the data will be embedded
    # into the graph as constants, which makes the training graph very
    # large and impractical. So use Dataset.from_generator() here instead,
    # but add prefetching and caching to improve performance.

    train_ds, _ = make_to_tensor(*datasets_from_numpy(sample_n=10000))

    return train_ds

  # Training progress is logged as INFO, so enable that logging level
  tf.logging.set_verbosity(tf.logging.INFO)

  t0 = time.time()
  ipu_estimator.train(input_fn=input_fn, steps=500000)
  t1 = time.time()

  duration_seconds = t1 - t0
  images_per_step = 10 * NUM_REPLICAS
  images_per_second = 100 * images_per_step / duration_seconds
  print("Took {:.2f} minutes, i.e. {:.0f} images per second".format(
      duration_seconds / 60, images_per_second))

def test(ipu_estimator):
  """Test the model on IPU by loading weights from the final checkpoint in the
  given `args.model_dir`."""

  def input_fn():
    # If using Dataset.from_tensor_slices(), the data will be embedded
    # into the graph as constants, which makes the training graph very
    # large and impractical. So use Dataset.from_generator() here instead,
    # but add prefetching and caching to improve performance.

    _, test_ds = make_to_tensor(*datasets_from_numpy(sample_n=1000))

    return test_ds

  num_steps = 2 * 1000 // (10 * NUM_REPLICAS)
  metrics = ipu_estimator.evaluate(input_fn=input_fn, steps=num_steps)
  test_loss = metrics["loss"]
  test_accuracy = metrics["accuracy"]

  print("Test loss: {:g}".format(test_loss))
  print("Test accuracy: {:.2f}%".format(100 * test_accuracy))

def main():
  est = create_ipu_estimator()
  train(est)
  test(est)


if __name__ == '__main__':
    main()