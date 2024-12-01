import datetime
import pickle
import time

import cupy as np

from src.add_ons.data_augmentation import DataAugmentation
from src.add_ons.early_stopping import EarlyStopping
from src.add_ons.weight_initialization import WeightInitialization
from src.add_ons.activation_function import ActivationFunction
from src.layers.activation_layer import ActivationLayer
from src.layers.convolution_2d_layer import Convolution2D
from src.layers.dropout_layer import DropoutLayer
from src.layers.flatten_layer import FlattenLayer
from src.layers.fully_connected_layer import FCLayer
from src.add_ons.learning_rate_schedulers import LearningRateScheduler
from src.add_ons.loss_function import LossFunction
from src.layers.max_pooling_2d_layer import MaxPoolingLayer2D
from src.network import Network
from src.add_ons.optimizers import Optimizer
from src.utils.read_data import read_data, to_categorical
from src.config import EPOCHS, BATCH_SIZE, LOG_FILE, LEARNING_RATE, CHANCE_OF_ALTERING_DATA, PATIENCE, MIN_DELTA_REL, \
    LEARNING_RATE_HALVE_AFTER


def main():
    # For debug:
    # np.seterr(all='raise')

    # Create log file if it does not exist already
    try:
        with open(LOG_FILE, 'x') as log_file:
            pass
    except:
        pass

    # Log new model
    string_to_be_logged = "--- --- --- --- --- NEW MODEL --- --- --- --- ---"
    print(string_to_be_logged)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(string_to_be_logged)

    # Read training and test data
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

    # Change model_to_load to filename in models/ to load a model. Otherwise, a new model will be trained.
    model_to_load = None
    model = None
    start_time = time.time()
    if model_to_load is None:
        # If there is no model to load, create a new neural network
        model = create_small_convolution_model()

        # Log hyper Parameters:
        string_to_be_logged = f"Hyperparameters: EPOCHS={model.epochs}, LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={model.batch_size}, LEARNING_RATE_SCHEDULER={model.learning_rate_scheduler}, LEARNING_RATE_HALVER_AFTER={model.learning_rate_halve_after}, \n CONVOLUTION_MODEL={model.convolution_network}, DATA_AUGMENTATION={model.data_augmentation is not None}, EARLY_STOPPING={model.early_stopping is not None}"
        if model.data_augmentation is not None:
            string_to_be_logged += f", CHANCE_OF_ALTERING_DATA={model.data_augmentation.chance_of_altering_data}"
        if model.early_stopping is not None:
            string_to_be_logged += f", PATIENCE={model.early_stopping.patience}, MIN_DELTA_REL={model.early_stopping.min_delta_rel}"
        string_to_be_logged += "\n"
        print(string_to_be_logged)
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(string_to_be_logged)

        model.set_loss_function(LossFunction.categorical_cross_entropy)
        model.fit(
            x_train,
            y_train,
        )

        # Save the model
        model_path = f"models/model_{datetime.datetime.now().strftime('%Y_%m_%dT%H:%M:%S')}_epochs_{EPOCHS}_learning_rate_{LEARNING_RATE}_batch_size_{BATCH_SIZE}.pkl"
        with open(model_path, "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(model, f)
    else:
        model_path = f"models/{model_to_load}"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    string_to_be_logged = "Total training time:" + "{:.2f}".format(elapsed_time_minutes) + "minutes"
    print(string_to_be_logged)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(string_to_be_logged + "\n")
    test_model(model, x_test, y_test)


def create_model() -> Network:
    model = Network()
    model.add_layer(FCLayer(28 * 28, 128, optimizer=Optimizer.Adam, weight_initialization=WeightInitialization.he_bias_zero))  # input_shape=(1, 28*28)    ;   output_shape=(1, 128)
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 128))
    model.add_layer(DropoutLayer(0.2, 128))

    model.add_layer(FCLayer(128, 50, optimizer=Optimizer.Adam, weight_initialization=WeightInitialization.he_bias_zero))  # input_shape=(1, 128)      ;   output_shape=(1, 50)
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 50))
    model.add_layer(DropoutLayer(0.2, 50))

    model.add_layer(FCLayer(50, 10, optimizer=Optimizer.Adam, weight_initialization=WeightInitialization.he_bias_zero))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
    model.add_layer(ActivationLayer(ActivationFunction.softmax, 10))

    # Set (hyper)parameters
    model.set_hyperparameters(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            learning_rate_scheduler=LearningRateScheduler.const,
            batch_size=BATCH_SIZE,
            data_augmentation=DataAugmentation(chance_of_altering_data=CHANCE_OF_ALTERING_DATA),
            early_stopping=EarlyStopping(patience=PATIENCE, min_delta_rel=MIN_DELTA_REL),
    )

    return model


def create_convolution_model() -> Network:
    model = Network()

    # Block 1: input_shape=(BATCH_SIZE, 1, 28, 28) output_shape=(BATCH_SIZE, 32, 28, 28)
    model.add_layer(Convolution2D(D_batch_size=BATCH_SIZE, C_number_channels=1, NF_number_of_filters=32, H_height_input=28, W_width_input=28, optimizer=Optimizer.Adam))
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 0, convolutional_network=True))
    model.add_layer(DropoutLayer(0.2, 0, convolutional_network=True))

    # Block 2: input_shape=(BATCH_SIZE, 32, 28, 28) output_shape=(BATCH_SIZE, 64, 14, 14)
    model.add_layer(Convolution2D(D_batch_size=BATCH_SIZE, C_number_channels=32, NF_number_of_filters=64, H_height_input=28, W_width_input=28, optimizer=Optimizer.Adam))
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 0, convolutional_network=True))
    model.add_layer(MaxPoolingLayer2D(D_batch_size=BATCH_SIZE, PS_pool_size=2, S_stride=2, C_number_channels=64, H_height_input=28, W_width_input=28))
    model.add_layer(DropoutLayer(0.2, 0, convolutional_network=True))

    # Block 3: input_shape=(BATCH_SIZE, 64, 14, 14) output_shape=(BATCH_SIZE, 96, 14, 14)
    model.add_layer(Convolution2D(D_batch_size=BATCH_SIZE, C_number_channels=64, H_height_input=14, W_width_input=14, NF_number_of_filters=96, optimizer=Optimizer.Adam))
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 0, convolutional_network=True))
    model.add_layer(DropoutLayer(0.2, 0, convolutional_network=True))

    # Block 4: input_shape=(BATCH_SIZE, 96, 14, 14) output_shape=(BATCH_SIZE, 128, 7, 7)
    model.add_layer(Convolution2D(D_batch_size=BATCH_SIZE, C_number_channels=96, H_height_input=14, W_width_input=14, NF_number_of_filters=128, optimizer=Optimizer.Adam))
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 0, convolutional_network=True))
    model.add_layer(MaxPoolingLayer2D(D_batch_size=BATCH_SIZE, PS_pool_size=2, S_stride=2, C_number_channels=128, H_height_input=14, W_width_input=14))
    model.add_layer(DropoutLayer(0.2, 0, convolutional_network=True))

    # Block 5: input_shape=(BATCH_SIZE, 128, 7, 7) output_shape=(BATCH_SIZE, 128 * 7 * 7)
    model.add_layer(FlattenLayer(D_batch_size=BATCH_SIZE, C_number_channels=128, H_height_input=7, W_width_input=7))

    # Block 6: input_shape=(BATCH_SIZE, 128 * 7 * 7) output_shape=(BATCH_SIZE, 10)
    model.add_layer(FCLayer(128 * 7 * 7, 10, optimizer=Optimizer.Adam, convolutional_network=True))
    model.add_layer(ActivationLayer(ActivationFunction.softmax, 10, convolutional_network=True))

    # Set (hyper)parameters
    model.set_hyperparameters(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        learning_rate_scheduler=LearningRateScheduler.const,
        batch_size=BATCH_SIZE,
        data_augmentation=DataAugmentation(chance_of_altering_data=CHANCE_OF_ALTERING_DATA),
        early_stopping=EarlyStopping(patience=PATIENCE, min_delta_rel=MIN_DELTA_REL),
        convolution_network=True,
    )

    return model


def create_small_convolution_model() -> Network:
    model = Network()

    # Block 1: input_shape=(BATCH_SIZE, 1, 28, 28) output_shape=(BATCH_SIZE, 8, 28, 28)
    model.add_layer(
        Convolution2D(D_batch_size=BATCH_SIZE, C_number_channels=1, NF_number_of_filters=8, H_height_input=28,
                      W_width_input=28, optimizer=None))
    # model.add_layer(BatchNormalization(D_batch_size=BATCH_SIZE, C_number_channels=8, H_height_input=28, W_width_input=28))
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 0, convolutional_network=True))
    model.add_layer(MaxPoolingLayer2D(D_batch_size=BATCH_SIZE, PS_pool_size=2, S_stride=2, C_number_channels=8,
                                      H_height_input=28, W_width_input=28))
    model.add_layer(DropoutLayer(0.2, 0, convolutional_network=True))

    # Block 2: input_shape=(BATCH_SIZE, 8, 28, 28) output_shape=(BATCH_SIZE, 16, 14, 14)
    model.add_layer(
        Convolution2D(D_batch_size=BATCH_SIZE, C_number_channels=8, NF_number_of_filters=16, H_height_input=14,
                      W_width_input=14, optimizer=None))
    # model.add_layer(BatchNormalization(D_batch_size=BATCH_SIZE, C_number_channels=16, H_height_input=14, W_width_input=14))
    model.add_layer(ActivationLayer(ActivationFunction.ReLu, 0, convolutional_network=True))
    model.add_layer(MaxPoolingLayer2D(D_batch_size=BATCH_SIZE, PS_pool_size=2, S_stride=2, C_number_channels=16,
                                      H_height_input=14, W_width_input=14))
    model.add_layer(DropoutLayer(0.2, 0, convolutional_network=True))

    # Block 3: input_shape=(BATCH_SIZE, 16, 7, 7) output_shape=(BATCH_SIZE, 16 * 7 * 7)
    model.add_layer(FlattenLayer(D_batch_size=BATCH_SIZE, C_number_channels=16, H_height_input=7, W_width_input=7))

    # Block 4: input_shape=(BATCH_SIZE, 128 * 7 * 7) output_shape=(BATCH_SIZE, 10)
    model.add_layer(FCLayer(16 * 7 * 7, 10, optimizer=None, convolutional_network=True))
    model.add_layer(ActivationLayer(ActivationFunction.softmax, 10, convolutional_network=True))

    # Set (hyper)parameters
    model.set_hyperparameters(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        learning_rate_scheduler=LearningRateScheduler.tunable,
        learning_rate_halve_after=LEARNING_RATE_HALVE_AFTER,
        batch_size=BATCH_SIZE,
        # data_augmentation=DataAugmentation(chance_of_altering_data=CHANCE_OF_ALTERING_DATA),
        early_stopping=EarlyStopping(patience=PATIENCE, min_delta_rel=MIN_DELTA_REL),
        convolution_network=True,
    )

    return model


def test_model(model: Network, x_test, y_test):
    if model.convolution_network:
        # Reshape data to be of shape (D, C, H, W) = (size_of_current_batch, 1, 28, 28) for convolution network
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    # Predict the output for the test data
    predictions = model.predict(x_test)
    predictions_flattened = predictions.reshape(len(predictions), 10)

    # Convert predictions to label indices
    predicted_labels = np.argmax(predictions_flattened, axis=1)
    actual_labels = np.argmax(y_test, axis=1)

    # Compare predicted labels with true labels
    correct_predictions = np.sum(predicted_labels == actual_labels)

    # Log the results
    string_to_be_logged = f"Number of correctly recognized images: {correct_predictions} out of {len(x_test)}"
    print(string_to_be_logged)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(string_to_be_logged + "\n")
    error = (len(x_test) - correct_predictions) / len(x_test)
    string_to_be_logged = f"The error rate is {error * 100}%"
    print(string_to_be_logged)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(string_to_be_logged + "\n")


if __name__ == "__main__":
    main()
