import keras
import keras.datasets.imdb as imdb
import math
import numpy as np
import tempfile
import tensorflow as tf
from typing import Dict, List

import utils

# IMDB data parameters
imdb_num_words = 2 ** 10 # Only these many words are kept
imdb_skip_top_words = 2 # Skip this many most freqently occurring words
imdb_num_classes = 2
imdb_max_seq_len = 64 # Discard all data with more than this # of words
    
# Model parameters
num_layers = 2
hidden_dimension = 8
embedding_dimension = 8
regularization = 1e-5
prob_output_name = 'prob'
mask_input_name = 'mask'
word_input_name = 'word'
hidden_input_names = ['hidden_input_%d' % idx for idx in range(num_layers)]
hidden_output_names = ['hidden_output_%d' % idx for idx in range(num_layers)]
cell_input_names = ['cell_input_%d' % idx for idx in range(num_layers)]
cell_output_names = ['cell_output_%d' % idx for idx in range(num_layers)]

# Training parameters
batch_size=32
epochs=5

def sentiment_model(batch_size: int = None, timesteps: int = None, training: bool = True):
    """
        Creates a model: embedding -> LSTMs -> FC -> sigmoid

        For training, we use variable batch size and time steps, along with a mask.

        For inference, we use a batch size = 1 and time steps = 1, with no mask.
    """

    # Input layers: one-hot word encoding, hidden features (for future timesteps), initial state
    initial_hiddens = [keras.layers.Input(
        batch_size=batch_size,
        shape=(hidden_dimension,),
        name=hidden_input_names[idx]
        ) for idx in range(num_layers)]
    initial_cells = [keras.layers.Input(
        batch_size=batch_size,
        shape=(hidden_dimension,), 
        name=cell_input_names[idx]
        ) for idx in range(num_layers)]
    words_one_hot = keras.layers.Input(
        batch_size=batch_size,
        shape=(timesteps, imdb_num_words), 
        name=word_input_name
    )
    mask = keras.layers.Input(
        batch_size=batch_size,
        shape=(timesteps,),
        name=mask_input_name
    ) if training else None

    # Use weight and activation regularization on all layers, to help with quantization
    regularizer = lambda: tf.keras.regularizers.L2(regularization)

    # Learned embedding layer to map words to features.
    # We use fully connected rather than keras.layers.Embedding because our model will have int8 inputs
    embedding = keras.layers.Dense(
        units=embedding_dimension,
        use_bias=False,
        activity_regularizer=regularizer(),
        kernel_regularizer=regularizer()
    )(words_one_hot)

    # Reshape the inputs to (timesteps, inputs)
    embedding = tf.keras.layers.Reshape(
        target_shape=(
            timesteps if timesteps is not None else -1, 
            embedding_dimension
        )
    )(embedding)

    # LSTM layers with initial state
    rnn_output = embedding
    final_hiddens = []
    final_cells = []
    for layer_idx, (initial_hidden, initial_cell) in enumerate(zip(initial_hiddens, initial_cells)):
        rnn_output, final_hidden, final_cell = keras.layers.RNN(
                keras.layers.LSTMCell(
                    units=hidden_dimension,
                    kernel_regularizer=regularizer()
                    ),
                return_sequences = True,
                return_state = True,
                unroll = not training
                )(rnn_output, mask=mask, initial_state=[initial_hidden, initial_cell])

        # Pass the hidden and cell states into identity layers, to rename them
        final_hiddens.append(keras.layers.Layer(name=hidden_output_names[layer_idx])(final_hidden))
        final_cells.append(keras.layers.Layer(name=cell_output_names[layer_idx])(final_cell))

    # Final classification layer
    final_rnn_output = rnn_output[:, -1]
    prob = keras.layers.Dense(
            units=imdb_num_classes, 
            activation='sigmoid',
            name=prob_output_name
            )(final_rnn_output)

    # Choose the model inputs/outputs for training and inference mode
    inputs=[words_one_hot] + initial_hiddens + initial_cells
    outputs={
        prob_output_name: prob
    }
    if training:
        inputs += [mask,]
    else:
        for idx, (hidden_output_name, cell_output_name) in enumerate(zip(hidden_output_names, cell_output_names)):
            outputs[hidden_output_name] = final_hiddens[idx]
            outputs[cell_output_name] = final_cells[idx]

    return keras.models.Model(
            inputs=inputs,
            outputs=outputs
    )

def prepare_inputs(x: List[int], y: List[int] = None):

    have_y = y is not None

    # Preprocessing: get a batch and pad to a common length
    max_seq_len = max(len(seq) for seq in x)
    x_words = np.zeros((len(x), max_seq_len))
    x_words.fill(np.nan)
    for x_idx, seq in enumerate(x):
        x_words[x_idx, :len(seq)] = seq
    mask = ~np.isnan(x_words)
    x_words[~mask] = 0

    # Convert words and labels to one-hot
    x_one_hot = keras.utils.to_categorical(x_words, imdb_num_words)
    y_one_hot = keras.utils.to_categorical(y, imdb_num_classes) if have_y else None

    # Initialize the hidden state to zeros
    initial_hiddens = {
            hidden_input_names[idx]: np.zeros((len(x), hidden_dimension), dtype=np.float32) for idx in range(num_layers)
            }
    initial_cells = {
            cell_input_names[idx]: np.zeros_like(initial_hiddens[hidden_input_names[idx]]) for idx in range(num_layers)
            }

    # Collect all the inputs
    inputs = {
        mask_input_name: mask,
        word_input_name: x_one_hot
    }
    inputs.update(initial_hiddens)
    inputs.update(initial_cells)

    return (inputs, y_one_hot) if have_y else inputs


def __process_sequences__(predict_fun, x, yield_inputs: bool = False) -> List[np.array]:
    """
        Runs the model on sequence data, then yields the inputs and initial 
        state as separate time slices.

        This can be used as a data generator for TF -> TFLite conversion, if y is None,
        or as an inference engine, if y is not None.

        Arguments:
            predict_fun: Runs prediction on the inputs. Same signature as predict() method of 
                keras.Model.
            x: Input data
            yield_intputs: If true, acts as a generator yielding model inputs. Used for TFLite quantization.
    """
    # Process each sequence individually
    sequence_inputs = prepare_inputs(x)
    batch_size, num_timesteps, num_words = sequence_inputs[word_input_name].shape
    print("Input stats:")
    print("\tNum sequences: %d" % batch_size)
    print("\tMax sequence length: %d" % num_timesteps)
    print("\tNum words: %d" % num_words)
    print("\tYielding inputs: %d" % yield_inputs)
    for batch_idx in range(batch_size):
        # Process each time step individually
        # For the first iteration, use the sequence input hidden state
        print("Generating sequence %d / %d..." % (batch_idx, batch_size))
        initial_hidden = [sequence_inputs[hidden_input_name][None, batch_idx] for hidden_input_name in hidden_input_names]
        initial_cell = [sequence_inputs[cell_input_name][None, batch_idx] for cell_input_name in cell_input_names]
        for timestep in range(num_timesteps):

            # Quit if the sequence is over
            mask_seq = sequence_inputs[mask_input_name][batch_idx]
            if not mask_seq[timestep]:
                break

            # Extract the inputs for this time slice
            inputs = {
                word_input_name: sequence_inputs[word_input_name][None, None, batch_idx, timestep],
            }
            for hidden_name, hidden_val in zip(hidden_input_names, initial_hidden):
                inputs[hidden_name] = hidden_val
            for cell_name, cell_val in zip(cell_input_names, initial_cell):
                inputs[cell_name] = cell_val

            # In quantization mode, return inputs to the TF -> TFlite converter
            if yield_inputs: 
                yield inputs

            # Proces the time step
            outputs = predict_fun(inputs)

            # Record the next hidden state
            initial_hidden = [outputs[hidden_output_name] for hidden_output_name in hidden_output_names]
            initial_cell = [outputs[cell_output_name] for cell_output_name in cell_output_names]

        # Collect the prediction at the end of each sequence
        if not yield_inputs:
            yield outputs[prob_output_name]

def __batch_process_sequences__(predict_fun, x: List) -> List:
    """
        Wrapper for process_sequences, using batch processing rather than a generator.
    """
    yield_inputs = False
    return [x for x in __process_sequences__(predict_fun, x, yield_inputs=yield_inputs)]

def __quantized_predict__(interpreter, signature_name: str, inputs: Dict, predict_fun) -> Dict:
    """
        Prediction function for quantized models. Can be used for TFLite or fmir, 
        depending on 'predict_fun'. Passed to __process_sequences__ during inference.
    """

    # Quantize inputs
    np_dtype = interpreter.get_input_details()[0]['dtype']
    runner = interpreter.get_signature_runner(signature_name)
    quantized_inputs = {name: utils.quantize_input_tensor(value, runner, name).astype(np_dtype) for name, value in inputs.items()} 

    # Predict
    quantized_outputs = predict_fun(quantized_inputs)

    # Dequantize outputs
    return {
        key: utils.dequantize_output_tensor(
            value,
            runner,
            key
        ).squeeze()
        for key, value in quantized_outputs.items()
    }

"""
    Main entry point. Trains a TF model, evaluates TF accuracy, 
    creates an inference version of the model, converts the inference model 
    to TFLite, and evaluates TFLite accuracy.
"""

# Get MNIST train and test splits
# From the training split, we will also take an 'eval' set 
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=imdb_num_words,
    skip_top=imdb_skip_top_words,
    maxlen=imdb_max_seq_len
)

# Abbreviate the test set, for speed 
num_test_samples = int(1e3)
x_test = x_test[:num_test_samples]
y_test = y_test[:num_test_samples]

# Create an network: embedding -> LSTM -> FC
train_model = sentiment_model(training=True)

# Train the model
train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
for epoch in range(epochs):
    print("Epoch %d / %d ..." % (epoch, epochs))

    rng = np.random.default_rng()
    num_batches = math.ceil(len(x_train) / batch_size)
    for batch_idx in range(num_batches):
        print("Batch %d / %d ..." % (batch_idx, num_batches))

        # Preprocess batch data
        batch_inds = rng.choice(len(x_train), size=batch_size) 
        x_batch = x_train[batch_inds]
        y_batch = y_train[batch_inds] 
        inputs, y_batch_one_hot = prepare_inputs(x_batch, y_batch)

        # Train on this batch
        train_model.train_on_batch(inputs, y_batch_one_hot)

# Evaluate on each split
train_inputs, y_train_one_hot = prepare_inputs(x_train, y_train)
test_inputs, y_test_one_hot = prepare_inputs(x_test, y_test)
tf_train_loss, tf_train_accuracy  = train_model.evaluate(train_inputs, y_train_one_hot, batch_size=batch_size)
tf_test_loss, tf_test_accuracy  = train_model.evaluate(test_inputs, y_test_one_hot, batch_size=batch_size)
print("Tensorflow loss:")
print("\tTrain: %f" % tf_train_loss)
print("\tTest: %f" % tf_test_loss)
print("Tensorflow accuracy:")
print("\tTrain: %f" % tf_train_accuracy)
print("\tTest: %f" % tf_test_accuracy)

# Truncate the datasets, for speed
num_quantization_sequences = 16
x_quant = x_train[:num_quantization_sequences]
y_quant = y_train[:num_quantization_sequences]

# Create an inference version of the model, for single batch size and time steps
inference_model = sentiment_model(batch_size=1, timesteps=1, training=False)
inference_model.set_weights(train_model.get_weights())

# Preprocess inference inputs
quant_inputs = prepare_inputs(x_quant)
test_inputs = prepare_inputs(x_test)

# Wrap the Keras model in a TF module, so we can get meaningful IO names
inference_module = utils.KerasModelModule(inference_model)
signature_name = inference_module.signature_name

# Convert to TFLite via a saved model
with tempfile.TemporaryDirectory() as dirname: #TODO use a set output dir
    # Save the TF module
    tf.saved_model.save(inference_module, dirname, signatures={signature_name: inference_module.predict_concrete})

    # Configure the TFLite converter for int8 quantization
    tf_dtype = tf.int8
    converter = tf.lite.TFLiteConverter.from_saved_model(dirname)
    converter.inference_input_type = tf_dtype 
    converter.inference_output_type = tf_dtype
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    keras_predict_fun = lambda inputs: inference_model.predict(inputs, verbose=False)
    converter.representative_dataset = lambda: __process_sequences__(keras_predict_fun, x_quant, yield_inputs=True) 

    # Convert the inference model to TFLite
    print("Converting TF model to TFLite...")
    tflite_flatbuffer = converter.convert()

# Create a prediction function to run the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_flatbuffer)
runner = interpreter.get_signature_runner(signature_name)
tflite_predict_fun = lambda inputs: __quantized_predict__(
    interpreter,
    signature_name,
    inputs,
    lambda quantized_inputs: runner(**quantized_inputs)
)

# Measure the TFLite model's accuracy
tflite_labels = []
reference_tflite_preds = []
print("Measuring TFLite accuracy...")
tflite_preds = __batch_process_sequences__(
    tflite_predict_fun, 
    x_test
)
tflite_accuracy = utils.classification_accuracy(tflite_preds, y_test)
print("TFLite reference accuracy: %f" % tflite_accuracy)

