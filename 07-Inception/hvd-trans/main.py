import os
import tensorflow as tf
import horovod.tensorflow as hvd
hvd_broadcast_done = False
hvd.init()
gpus = tf.config.experimental.list_physical_devices("GPU", )
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True, )
if gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU", )
import numpy as np
from tensorflow import keras

# tensorboard
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/hvd-trans-board-manual/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)



tf.random.set_seed(22, )
np.random.seed(22, )
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
assert tf.__version__.startswith("2.", )
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
(x_train, x_test) = (x_train.astype(np.float32, ) / 255.0, x_test.astype(np.float32, ) / 255.0)
(x_train, x_test) = (np.expand_dims(x_train, axis=3, ), np.expand_dims(x_test, axis=3, ))
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train), ).batch(256, )
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test), ).batch(256, )
if hvd.rank() == 0:
  print(x_train.shape, y_train.shape, )
if hvd.rank() == 0:
  print(x_test.shape, y_test.shape, )
class ConvBNRelu(keras.Model, ):
  def __init__(self, ch, kernelsz=3, strides=1, padding="same", ):
    super(ConvBNRelu, self, ).__init__()
    self.model = keras.models.Sequential([keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding, ), keras.layers.BatchNormalization(), keras.layers.ReLU()], )
  def call(self, x, training=None, ):
    x = self.model(x, training=training, )
    return x
class InceptionBlk(keras.Model, ):
  def __init__(self, ch, strides=1, ):
    super(InceptionBlk, self, ).__init__()
    self.ch = ch
    self.strides = strides
    self.conv1 = ConvBNRelu(ch, strides=strides, )
    self.conv2 = ConvBNRelu(ch, kernelsz=3, strides=strides, )
    self.conv3_1 = ConvBNRelu(ch, kernelsz=3, strides=strides, )
    self.conv3_2 = ConvBNRelu(ch, kernelsz=3, strides=1, )
    self.pool = keras.layers.MaxPooling2D(3, strides=1, padding="same", )
    self.pool_conv = ConvBNRelu(ch, strides=strides, )
  def call(self, x, training=None, ):
    x1 = self.conv1(x, training=training, )
    x2 = self.conv2(x, training=training, )
    x3_1 = self.conv3_1(x, training=training, )
    x3_2 = self.conv3_2(x3_1, training=training, )
    x4 = self.pool(x, )
    x4 = self.pool_conv(x4, training=training, )
    x = tf.concat([x1, x2, x3_2, x4], axis=3, )
    return x
class Inception(keras.Model, ):
  def __init__(self, num_layers, num_classes, init_ch=16, **kwargs, ):
    super(Inception, self, ).__init__(**kwargs, )
    self.in_channels = init_ch
    self.out_channels = init_ch
    self.num_layers = num_layers
    self.init_ch = init_ch
    self.conv1 = ConvBNRelu(init_ch, )
    self.blocks = keras.models.Sequential(name="dynamic-blocks", )
    for block_id in range(num_layers, ):
      for layer_id in range(2, ):
        if layer_id == 0:
          block = InceptionBlk(self.out_channels, strides=2, )
        else:
          block = InceptionBlk(self.out_channels, strides=1, )
        self.blocks.add(block, )
      self.out_channels *= 2
    self.avg_pool = keras.layers.GlobalAveragePooling2D()
    self.fc = keras.layers.Dense(num_classes, )
  def call(self, x, training=None, ):
    out = self.conv1(x, training=training, )
    out = self.blocks(out, training=training, )
    out = self.avg_pool(out, )
    out = self.fc(out, )
    return out
batch_size = 32
epochs = 100
model = Inception(2, 10, )
model.build(input_shape=(None, 28, 28, 1), )
if hvd.rank() == 0:
  model.summary()
optimizer = keras.optimizers.Adam(learning_rate=0.001 * hvd.size(), )
criteon = keras.losses.CategoricalCrossentropy(from_logits=True, )
acc_meter = keras.metrics.Accuracy()
for epoch in range(100, ):
  for (step, (x, y)) in enumerate(db_train, ):
    with tf.GradientTape() as tape:
      logits = model(x, )
      loss = criteon(tf.one_hot(y, depth=10, ), logits, )
    tape = hvd.DistributedGradientTape(tape, )
    grads = tape.gradient(loss, model.trainable_variables, )
    id_new = zip(grads, model.trainable_variables, )
    optimizer.apply_gradients(id_new, )
    #global hvd_broadcast_done
    if not hvd_broadcast_done:
      # manual : trainable_variables -> variables
      hvd.broadcast_variables(model.variables, root_rank=0, )
      hvd.broadcast_variables(optimizer.variables(), root_rank=0, )
      hvd_broadcast_done = True
    if step % 10 == 0:
      if hvd.rank() == 0:
        print(epoch, step, "loss:", loss.numpy(), )
  acc_meter.reset_states()
  for (x, y) in db_test:
    logits = model(x, training=False, )
    pred = tf.argmax(logits, axis=1, )
    acc_meter.update_state(y, pred, )

  # tensorboard
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', loss, step=100*epoch+step)
    tf.summary.scalar('acc', acc_meter.result(), step=100*epoch+step)

  if hvd.rank() == 0:
    print(epoch, "evaluation acc:", acc_meter.result().numpy(), )
