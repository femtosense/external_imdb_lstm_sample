"""
This file contains some utilities for handling Keras, TFlite, and ML data in general.
"""

import keras
import numpy as np
import tensorflow as tf


def classification_accuracy(logits: np.array, labels: np.array):
    predictions = np.argmax(logits, axis=1)
    return np.mean(labels == predictions)

def quantize_input_tensor(x: int, runner, name: str):
    """
    Quantize an input tensor from float to int, using metadata from the TFLite
    SignatureRunner.
    """
    details = runner.get_input_details()[name]
    scale, zero_point = details['quantization']
    return (np.array(x, dtype=float) / scale + zero_point).astype(int)

def dequantize_output_tensor(x: int, runner, name: str):
    """
        Dequantize an output value back to a float, using metadata from the TFLite 
        SignatureRunner.
    """
    details = runner.get_output_details()[name]
    scale, zero_point = details['quantization']
    return scale * (np.array(x, dtype=float) - zero_point)

class KerasModelModule(tf.Module):
    """
        Wrap a Keras model in a tf.Module, so we can provide meaningful
        input/output names to the signature runner.
    """
    def __init__(self, model: keras.Model):
        self.model = model

        # Prediction function with input signature
        self.predict_concrete = self.__predict__.get_concrete_function([
            tf.TensorSpec(shape=x.shape, name=x.name) 
            for x in model.input_spec
        ])

    @tf.function
    def __predict__(self, inputs):
        return self.model(inputs)

    @property
    def signature_name(self):
        return 'predict'

