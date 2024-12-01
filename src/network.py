import time

import numpy as np
import random
from nptyping.ndarray import NDArray

from src.add_ons.data_augmentation import DataAugmentation
from src.add_ons.early_stopping import EarlyStopping
from src.layers.layer import Layer
from src.add_ons.learning_rate_schedulers import LearningRateScheduler
from src.add_ons.loss_function import LossFunction
from src.config import LOG_FILE, DEBUGGING
from src.utils.utils import shuffle_in_unison, create_batches


class Network:
    """
    Class representing a neural network. The network can be set up using self.add() to add layers and
    self.loss_function() to set the loss function to use. The network can be trained using self.fit().
    """
    layers: list[Layer]
    loss_function: LossFunction | None
    convolution_network: bool = False

    def __init__(self):
        self.layers: list[Layer] = []
        self.loss_function: LossFunction | None = None
        # Hyperparameters
        self.learning_rate: float | None = None
        self.learning_rate_scheduler: LearningRateScheduler | None = None
        self.epochs: int | None = None
        self.batch_size: int | None = None
        self.data_augmentation: DataAugmentation | None = None
        self.early_stopping: EarlyStopping | None = None

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

    def set_hyperparameters(
            self,
            learning_rate: float,
            learning_rate_scheduler: LearningRateScheduler = LearningRateScheduler.const,
            epochs: int = 100,
            batch_size: int = 1,
            data_augmentation: DataAugmentation | None = None,
            early_stopping: EarlyStopping | None = None,
            convolution_network: bool = False
    ):
        """
        Set the hyperparameters for training the network.
        :param learning_rate: Learning rate to use for training.
        :param learning_rate_scheduler: Learning rate scheduler to use. Depending on which scheduler is used,
        might use or overwrite the learning rate parameter. Defaults to const.
        :param epochs: Number of epochs to train the network for. Defaults to 100.
        :param batch_size: Size of the batches to use for training. Defaults to 1 for stochastic gradient descent.
        :param data_augmentation: Optional data augmentation to use for training.
        It is applied to each batch before training. Defaults to None.
        :param early_stopping: Optional early stopping to use for training. Defaults to None.
        :param convolution_network: Whether the network is a convolutional network. Defaults to False.
        """
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.early_stopping = early_stopping
        self.convolution_network = convolution_network

    def predict(self, input_data: NDArray) -> NDArray:
        """
        Predict the output for a given input data.
        :param input_data: Input data to predict the output for.
        :return: A list containing the predicted output for each input data sample. Each prediction is a numpy array
        with the probability for each possible label.
        """
        if self.convolution_network:
            # Additional 1 for the batch size or channel dimension
            input_data = input_data.reshape(input_data.shape[0], 1, 1, 28, 28)
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.predict(output, 1)
            result.append(output)

        return np.array(result)

    def fit(
            self,
            x_train: NDArray,
            y_train: NDArray,
    ):
        """
        Train the network on the given training data.
        :param x_train: Training data for the network.
        :param y_train: Training labels for the network.
        """
        number_of_samples = len(x_train)
        learning_rate = self.learning_rate

        # training loop
        for epoch_index in range(self.epochs):
            start_time = time.time()

            # Shuffle data and create batches
            if not DEBUGGING:
                x_train, y_train = shuffle_in_unison(x_train, y_train)
            if self.convolution_network:
                # Reshape data to be of shape (D, C, H, W) = (size_of_current_batch, 1, 28, 28) for convolution network
                x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
            x_train_batches, y_train_batches = create_batches(x_train, y_train, self.batch_size)

            # Set learning rate for this epoch
            learning_rate = self.learning_rate_scheduler.get_learning_rate(learning_rate, epoch_index)

            # Error of the epoch to be displayed
            err = 0
            for current_batch_index in range(len(x_train_batches)):
                if self.data_augmentation is not None:
                    # Apply data augmentation with a certain chance
                    if random.random() < self.data_augmentation.chance_of_altering_data:
                        # Apply data augmentation
                        x_train_batches[current_batch_index] = self.data_augmentation.batch_apply(x_train_batches[current_batch_index])

                size_of_current_batch = len(x_train_batches[current_batch_index])
                # Initialize matrix to save vectors containing the error to propagate for each sample in the batch
                # batch_error_to_propagate[i] contains the error for sample i in the batch
                batch_error_to_propagate: NDArray = np.zeros((size_of_current_batch, 1, 10))

                if not self.convolution_network:
                    for current_sample_index in range(size_of_current_batch):
                        # forward propagation
                        output = x_train_batches[current_batch_index][current_sample_index]
                        for layer in self.layers:
                            output = layer.forward_propagation(output, size_of_current_batch, current_sample_index)

                        # compute loss (for display purpose only)
                        error_test = self.loss_function.function(y_train_batches[current_batch_index][current_sample_index], output)
                        err += self.loss_function.function(y_train_batches[current_batch_index][current_sample_index], output)
                        batch_error_to_propagate[current_sample_index] = self.loss_function.derivative(y_train_batches[current_batch_index][current_sample_index], output)

                else:
                    # forward propagation
                    output = x_train_batches[current_batch_index]
                    for layer in self.layers:
                        output = layer.forward_propagation(output, size_of_current_batch, current_batch_index)

                    # compute loss (for display purpose only)
                    error_test = self.loss_function.function(y_train_batches[current_batch_index], output)
                    err += self.loss_function.function(y_train_batches[current_batch_index], output)
                    batch_error_to_propagate = self.loss_function.derivative(y_train_batches[current_batch_index], output)

                # backward propagation
                for layer in reversed(self.layers):
                    batch_error_to_propagate = layer.backward_propagation(batch_error_to_propagate, learning_rate, epoch_index + 1)

            # calculate average error on all samples
            err /= number_of_samples
            end_time = time.time()
            elapsed_time = end_time - start_time
            string_to_be_logged = (f"epoch {epoch_index + 1}" + " " * (len(str(self.epochs)) - len(str(epoch_index + 1))) + f"/{self.epochs}   "
                                   + "time: " + "{:.2f}".format(elapsed_time) + "s   "
                                   + "error=" + "{:.2e}".format(err) + "   "
                                   + "learning rate=" + "{:.2e}".format(learning_rate))
            print(string_to_be_logged)
            with open(LOG_FILE, 'a') as log_file:
                log_file.write(string_to_be_logged + "\n")

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping.monitor == "val_loss":
                    if self.early_stopping.should_stop(err, self.layers, epoch_index + 1):
                        break
