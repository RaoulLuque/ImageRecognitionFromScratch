import numpy as np

from src.activation_function import ActivationFunction
from src.activation_layer import ActivationLayer
from src.fully_connected_layer import FCLayer
from src.learning_rate_schedulers import LearningRateScheduler
from src.loss_function import LossFunction
from src.network import Network
from src.read_data import read_data, to_categorical
from src.config import EPOCHS, BATCH_SIZE


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
    model = create_model()

    # Train only on part of the data since all of it would be pretty slow since batches are not implemented yet
    model.set_loss_function(LossFunction.mse)
    model.fit(x_train, y_train, epochs=EPOCHS, learning_rate_scheduler=LearningRateScheduler.const, batch_size=BATCH_SIZE)

    test_model(model, x_test, y_test)


def create_model() -> Network:
    model = Network()
    model.add_layer(FCLayer(28 * 28, 128))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
    model.add_layer(ActivationLayer(ActivationFunction.tanh, 128))
    # model.add_layer(FCLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
    # model.add_layer(ActivationLayer(ActivationFunction.tanh, 50))
    model.add_layer(FCLayer(128, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
    model.add_layer(ActivationLayer(ActivationFunction.tanh, 10))
    return model


def test_model(model: Network, x_test, y_test):
    predictions = model.predict(x_test)
    predictions_flattened = np.array(predictions).reshape(len(predictions), 10)

    # Convert predictions to label indices
    predicted_labels = np.argmax(predictions_flattened, axis=1)
    actual_labels = np.argmax(y_test, axis=1)

    # Compare predicted labels with true labels
    correct_predictions = np.sum(predicted_labels == actual_labels)

    print(f"Number of correctly recognized images: {correct_predictions} out of {len(x_test)}")
    error = (len(x_test) - correct_predictions) / len(x_test)
    print(f"The error rate is {error * 100}%")


if __name__ == "__main__":
    main()
