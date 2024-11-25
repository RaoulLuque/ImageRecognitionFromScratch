from src.activation_function import ActivationFunction
from src.activation_layer import ActivationLayer
from src.fully_connected_layer import FCLayer
from src.loss_function import LossFunction
from src.network import Network
from src.read_data import read_data, to_categorical


def main():
    (x_train, y_train, x_test, y_test) = read_data()

    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # Convert labels into one-hot encoding
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = to_categorical(y_test)

    # Create neural network
    model = Network()
    model.add_layer(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
    model.add_layer(ActivationLayer(ActivationFunction.tanh))
    model.add_layer(FCLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
    model.add_layer(ActivationLayer(ActivationFunction.tanh))
    model.add_layer(FCLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
    model.add_layer(ActivationLayer(ActivationFunction.tanh))

    # Train only on part of the data since all of it would be pretty slow since batches are not implemented yet
    model.set_loss_function(LossFunction.mse)
    model.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)


if __name__ == "__main__":
    main()
