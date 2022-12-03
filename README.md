# TFLite stateless unrolled LSTM sample

Example of training a Keras LSTM model on the IMDB dataset, with variable sequence length and batch size, then converting to a much simpler reperesentation of the model for inference accelerators.

The inference version of the model is unrolled by Keras, with a batch size and time step of 1, so it avoids many of the trickier aspects of RNNs. Specifically, it has neither state, nor control flow, nor dynamic tensor sizes. All state is explicitly managed by the user as input/output tensors.

The model in quantized to 8x8 mode (8-bit weights, 8-bit activations).

This code is provided free of use to anyone, with the hope of advancing industry benchmarks for ML inference accelerators.

For questions or comments, contact Blaine Rister (blaine.rister@femtosense.ai).

## Usage

First install the requirements with `pip install -r requirements.txt`.

Then, run the program with `python run_imdb_example.py`.

The program prints a log showing the accuracy of the original TF model and the quantized TFLite version.
