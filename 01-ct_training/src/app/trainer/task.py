import tensorflow as tf
import argparse
import os
import sys, traceback
from google.cloud.aiplatform.training_utils import cloud_profiler

"""Train an mnist model and use cloud_profiler for profiling."""

def _create_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


def main(args):
    print('Initialize the profiler ...')
    
    cloud_profiler.init()
    print('The profiler initiated.')

    print('Loading and preprocessing data ...')
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print('Creating and training model ...')

    model = _create_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )

    log_dir = "logs"
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR']
        
    print(f'log_dir: {log_dir}')

    print('Setting up the TensorBoard callback ...')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    print('Training model ...')
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        verbose=2,
        callbacks=[tensorboard_callback],
    )
    print('Training completed.')

    print('Saving model ...')

    model_dir = "model"
    if 'AIP_MODEL_DIR' in os.environ:
        model_dir = os.environ['AIP_MODEL_DIR']
        
    print(f'model_dir: {model_dir}')
    
    tf.saved_model.save(model, model_dir)

    print('Model saved at ' + model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to run model."
    )

    args = parser.parse_args()
    main(args)