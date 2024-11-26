import time

import numpy as np
from nptyping.ndarray import NDArray

from src.layers.layer import Layer
from src.add_ons.learning_rate_schedulers import LearningRateScheduler
from src.add_ons.loss_function import LossFunction
from src.config import LEARNING_RATE, LOG_FILE
from src.utils.utils import shuffle_in_unison, create_batches


class Network:
    """
    Class representing a neural network. The network can be set up using self.add() to add layers and
    self.loss_function() to set the loss function to use. The network can be trained using self.fit().
    """
    layers: list[Layer]
    loss_function: LossFunction | None

    def __init__(self):
        self.layers: list[Layer] = []
        self.loss_function: LossFunction | None = None

    def add_layer(self, layer: Layer):
        """
        Add a layer to the network. The layer is added at the end of the network (appended to the network).
        :param layer: Layer to add to the network.
        """
        self.layers.append(layer)

    def set_loss_function(self, loss_function: LossFunction):
        """
        Set the loss function to use for training the network.
        :param loss_function: Loss function to use.
        """
        self.loss_function = loss_function

    def predict(self, input_data: NDArray) -> list[NDArray]:
        """
        Predict the output for a given input data.
        :param input_data: Input data to predict the output for.
        :return: A list containing the predicted output for each input data sample. Each prediction is a numpy array
        with the probability for each possible label.
        """
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.predict(output)
            result.append(output)

        return result

    def fit(self, x_train: NDArray, y_train: NDArray, epochs: int, learning_rate_scheduler: LearningRateScheduler, batch_size: int = 1):
        """
        Train the network on the given training data.
        :param x_train: Training data for the network.
        :param y_train: Training labels for the network.
        :param epochs: Number of epochs to train the network for.
        :param learning_rate_scheduler: Learning rate scheduler to be used by the network.
        :param batch_size: Size of the batches to use for training. Defaults to 1 (stochastic descent)
        """
        number_of_samples = len(x_train)
        learning_rate = LEARNING_RATE

        # training loop
        for epoch_index in range(epochs):
            start_time = time.time()

            # Shuffle data and create batches
            x_train, y_train = shuffle_in_unison(x_train, y_train)
            x_train_batches, y_train_batches = create_batches(x_train, y_train, batch_size)

            # Set learning rate for this epoch
            learning_rate = learning_rate_scheduler.get_learning_rate(learning_rate, epoch_index)

            # Error of the epoch to be displayed
            err = 0
            for current_batch_index in range(len(x_train_batches)):
                size_of_current_batch = len(x_train_batches[current_batch_index])
                # Initialize matrix to save vectors containing the error to propagate for each sample in the batch
                # batch_error_to_propagate[i] contains the error for sample i in the batch
                batch_error_to_propagate: NDArray = np.zeros((size_of_current_batch, 1, 10))
                for current_sample_index in range(size_of_current_batch):
                    # forward propagation
                    output = x_train_batches[current_batch_index][current_sample_index]
                    for layer in self.layers:
                        output = layer.forward_propagation(output, size_of_current_batch, current_sample_index)

                    # compute loss (for display purpose only)
                    err += self.loss_function.function(y_train_batches[current_batch_index][current_sample_index], output)
                    batch_error_to_propagate[current_sample_index] = self.loss_function.derivative(y_train_batches[current_batch_index][current_sample_index], output)

                # backward propagation
                for layer in reversed(self.layers):
                    batch_error_to_propagate = layer.backward_propagation(batch_error_to_propagate, learning_rate, epoch_index + 1)

            # calculate average error on all samples
            err /= number_of_samples
            end_time = time.time()
            elapsed_time = end_time - start_time
            string_to_be_logged = (f"epoch {epoch_index + 1}" + " " * (len(str(epochs)) - len(str(epoch_index + 1))) + f"/{epochs}   "
                                   + "time: " + "{:.2f}".format(elapsed_time) + "s   "
                                   + "error=" + "{:.4f}".format(err) + "   "
                                   + "learning rate=" + "{:.4f}".format(learning_rate))
            print(string_to_be_logged)
            with open(LOG_FILE, 'a') as log_file:
                log_file.write(string_to_be_logged + "\n")
