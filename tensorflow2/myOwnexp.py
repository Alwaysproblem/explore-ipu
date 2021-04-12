import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf2
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.python import ipu
from functools import partial


cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

sample_n = 10000
meana = np.array([1, 1])
cova = np.array([[0.1, 0],[0, 0.1]])

meanb = np.array([2, 2])
covb = np.array([[0.1, 0],[0, 0.1]])

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

def lr(input_dim = 2, output_dim = 1, hidden = 32):
    inputs = Input(name="data", shape=(input_dim,))
    lr_l = layers.Dense(hidden, activation="relu", name = "linear", )(inputs)
    outputs = layers.Dense(output_dim, 
            activation='sigmoid', use_bias=True)(lr_l)
    
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)
    # model = ipu.keras.Model(inputs=inputs, outputs=outputs)

    return model

ds = tf2.data.Dataset.from_tensor_slices((X, y))
ds = ds.batch(5, drop_remainder=True).shuffle(5)
# ds_x = tf2.data.Dataset.from_tensor_slices(X)
# ds_y = tf2.data.Dataset.from_tensor_slices(y)

# ds = tf2.data.Dataset.zip((ds_x, ds_y))
ds = ds.repeat()

for xt, yt in ds.take(1):
  print(xt)
  print(yt)

infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(ds, feed_name="infeed")
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

infeed_queue.initializer
print(infeed_queue._dequeue())


def training_step(model, opt, count, X, y):
  with tf2.GradientTape() as tape:
    logits = model(X, training=True)
    losses = tf2.math.reduce_mean(tf2.keras.losses.binary_crossentropy(y, logits))
  
  grads = tape.gradient(losses, model.trainable_variables)
  opt.apply_gradients(zip(grads, model.trainable_variables))
  outfeed_queue.enqueue({"losses": losses})
  return count


@tf2.function(experimental_compile=True)
def my_train_loop():
  model = lr()
  opt = tf2.keras.optimizers.Adam()
  counter = 0
  training_step_with_model = partial(training_step, model, opt)
  count = ipu.loops.repeat(10, training_step_with_model, [counter], infeed_queue)
  return count


# Initialize the IPU default strategy.
strategy = ipu.ipu_strategy.IPUStrategy()

with strategy.scope():
  infeed_queue.initializer
  losses = strategy.experimental_run_v2(my_train_loop)
  print("losses", losses)

  # The outfeed dequeue has to happen after the outfeed enqueue op has been executed.
  result = outfeed_queue.dequeue()

  print("outfeed result", result)


# # Create an IPU distribution strategy
# strategy = ipu.ipu_strategy.IPUStrategy()

# with strategy.scope():
#   # An optimizer for updating the trainable variables
#     opt = tf2.keras.optimizers.SGD(0.01)

#   # Create an instance of the model
#     model = lr()

#   # Train the model
#     for i in range(5):
#         loss = strategy.experimental_run_v2(training_step, args=[X, y, model, opt])
#         print("Step " + str(i) + " loss = " + str(loss.numpy()))

# strategy = ipu.ipu_strategy.IPUStrategy()

# with strategy.scope():
#     model = lr()
#     model.summary()

#     model.compile('adam', loss=tf2.losses.BinaryCrossentropy())
#     model.fit(X, y, epochs=5, batch_size = 10)

#     print(model.predict(X[0: 10], batch_size = 5))
