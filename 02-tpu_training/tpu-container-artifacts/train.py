# Single, Mirror and Multi-Machine Distributed Training for CIFAR-10

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys, traceback

# gcp
from google.cloud import aiplatform
from google.cloud.aiplatform.training_utils import cloud_profiler

tfds.disable_progress_bar()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr',
                    default=0.01, type=float,
                    help='Learning rate.')
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')
parser.add_argument('--steps', dest='steps',
                    default=200, type=int,
                    help='Number of steps per epoch.')
parser.add_argument('--distribute', dest='distribute', type=str, default='single',
                    help='distributed training strategy')
parser.add_argument('--batch_size',default=128, 
                    type=int, help='non-global')
parser.add_argument('--project', type=str)
parser.add_argument('--location', dest='location',
                    default="us-central1", type=str,)
parser.add_argument('--tb_instance', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--experiment_run', type=str)

args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Single Machine, single compute device
if args.distribute == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple TPU devices
elif args.distribute == 'tpu':
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
# Single Machine, multiple compute device
elif args.distribute == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif args.distribute == 'multi':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Multi-worker configuration
print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

# Initialize the profiler.
print('Initialize the profiler ...')
try:
    cloud_profiler.init()
except:
    ex_type, ex_value, ex_traceback = sys.exc_info()
    print("*** Unexpected:", ex_type.__name__, ex_value)
    traceback.print_tb(ex_traceback, limit=10, file=sys.stdout)
print('The profiler initiated.')

# initialize Vertex AI sdk
aiplatform.init(
    project=args.project, 
    location=args.location,
    experiment=args.experiment_name,
    experiment_tensorboard=args.tb_instance,
)

# set job directories
MODEL_DIR = os.getenv("AIP_MODEL_DIR")
print(f"MODEL_DIR = {MODEL_DIR}")

log_dir = "logs"
if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
    log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR']
print(f"log_dir = {log_dir}")

print('Setting up the TensorBoard callback ...')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    update_freq="epoch",
    histogram_freq=1,
    # embeddings_freq=1,
    # profile_batch=20,
    # write_graph=True,
)

# Preparing dataset
BUFFER_SIZE = 10000
# BATCH_SIZE = 64

def make_datasets_unbatched():
    # Scaling CIFAR10 data from (0, 255] to (0., 1.]
    
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    datasets, info = tfds.load(
        name='cifar10', with_info=True, as_supervised=True
    )
    
    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE).repeat()


# Build the Keras model
def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    )
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr),
        metrics=['accuracy']
    )
    return model

# Train the model
NUM_WORKERS = strategy.num_replicas_in_sync
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
GLOBAL_BATCH_SIZE = args.batch_size * NUM_WORKERS

print(f"NUM_WORKERS = {NUM_WORKERS}")
print(f"BATCH_SIZE  = {args.batch_size}")
print(f"GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}")

train_dataset = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)

with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    model = build_and_compile_cnn_model()

model.fit(
    x=train_dataset, 
    epochs=args.epochs, 
    steps_per_epoch=args.steps,
    callbacks=[tensorboard_callback],
)

if args.distribute=="tpu":
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model.save(MODEL_DIR, options=save_locally)
else:
    model.save(MODEL_DIR)
    
# print('uploading TB logs ...')
# aiplatform.upload_tb_log(
#     tensorboard_experiment_name=args.experiment_name,
#     logdir=log_dir,
#     run_name_prefix=f"{args.experiment_run}-",
#     allowed_plugins=["profile"],
# )
