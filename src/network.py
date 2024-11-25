from nptyping.ndarray import NDArray

from src.layer import Layer
from src.loss_function import LossFunction
from src.optimizations import shuffle_in_unison


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
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        """
        Train the network on the given training data.
        :param x_train: Training data for the network.
        :param y_train: Training labels for the network.
        :param epochs: Number of epochs to train the network for.
        :param learning_rate: Learning rate to be used to training the network.
        :return:
        """
        number_of_samples = len(x_train)

        # training loop
        for i in range(epochs):
            x_train, y_train = shuffle_in_unison(x_train, y_train)

            err = 0
            for j in range(number_of_samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss_function.function(y_train[j], output)

                # backward propagation
                error = self.loss_function.derivative(y_train[j], output)
                # Iterate the layers backward because error is propagated from the last layer to the first
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= number_of_samples
            print(f"epoch {i+1}" + " " * (len(str(epochs)) - len(str(i+1))) +
                  f"/{epochs}    error=" + "{:.12f}".format(err) + f" learning rate={learning_rate}")
