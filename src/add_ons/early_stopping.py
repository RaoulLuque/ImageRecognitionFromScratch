import cupy as np
from nptyping import NDArray

from src.config import EPSILON, LOG_FILE
from src.layers.layer import Layer


class EarlyStopping:
    """
    Class for early stopping.
    """

    def __init__(self, patience: int = 10, min_delta_rel: float = 0.0, restore_best_weights: bool = True, monitor: str = "val_loss"):
        """
        Initialize the early stopping object.
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param min_delta_rel: Minimum relative change in the monitored quantity to qualify as an improvement.
        :param restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity.
        :param monitor: Quantity/Metric to be monitored.
        """
        self.patience: int = patience
        self.min_delta_rel: float = min_delta_rel
        self.restore_best_weights: bool = restore_best_weights
        self.monitor: str = monitor
        # Metric values and weights will always be kept at a same length of patience
        self.metric_values: NDArray = np.array([])
        self.weights: list = []

    def should_stop(self, metric_value_current_epoch: float, layers_current_epoch: list[Layer], epoch: int) -> bool:
        """
        Check if the early stopping condition is met and reapply best weights if restore_best_weights is True.
        :param metric_value_current_epoch: Current value of the metric value that is being tracked.
        :param layers_current_epoch: Layers of the current epoch.
        :param epoch: Current epoch. Used for printing information.
        :return: True if the early stopping condition is met, False otherwise.
        """
        # Update metric values and weights
        weight_list = extract_weights(layers_current_epoch)
        self.__update(metric_value_current_epoch, weight_list)
        if len(self.metric_values) < self.patience:
            return False

        # Check if it should be stopped
        if self.__is_stopping_condition_fulfilled():
            if self.restore_best_weights:
                best_weight_index = np.argmin(self.metric_values)
                best_weights = self.weights[best_weight_index]
                apply_weights(layers_current_epoch, best_weights)
                # Log early stopping
                string_to_be_logged = f"Early stopping: Restoring best weights from epoch {epoch - self.patience + best_weight_index + 1}"
                print(string_to_be_logged)
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(string_to_be_logged + "\n")
                return True
            # Log early stopping
            string_to_be_logged = f"Early stopping"
            print(string_to_be_logged)
            with open(LOG_FILE, 'a') as log_file:
                log_file.write(string_to_be_logged + "\n")
            return True
        return False

    def __update(self, metric_current_epoch: float, weights_current_epoch: list) -> None:
        """
        Update the metric values and weights.
        :param metric_current_epoch: Metric value of the current epoch.
        :param weights_current_epoch: Weights of the current epoch.
        """
        if len(self.metric_values) < self.patience:
            self.metric_values = np.append(self.metric_values, metric_current_epoch)
            self.weights.append(weights_current_epoch)
        else:
            self.metric_values = np.append(self.metric_values[1:], metric_current_epoch)
            self.weights = self.weights[1:]
            self.weights.append(weights_current_epoch)

    def __is_stopping_condition_fulfilled(self) -> bool:
        if self.metric_values[0] <= np.min(self.metric_values) * (1 + self.min_delta_rel):
            return True
        return False


def extract_weights(layers_current_epoch: list[Layer]) -> list:
    """
    Extract weights from layers.
    :param layers_current_epoch: Layers of the current epoch.
    :return: List of tuples of weights and bias of each layer that has weights in order of layers_current_epoch.
    """
    weights: list = []
    for layer in layers_current_epoch:
        if hasattr(layer, 'weights'):
            weights.append((layer.weights.copy(), layer.bias.copy()))
        else:
            weights.append(None)
    return weights


def apply_weights(layers: list[Layer], weights: list) -> list[Layer]:
    """
    Apply weights to layers.
    :param layers: Layers to apply weights to.
    :param weights: Weights to apply to layers.
    :return: Layers with weights applied.
    """
    for index, layer in enumerate(layers):
        if hasattr(layer, 'weights'):
            layer.weights = weights[index][0]
            layer.bias = weights[index][1]
    return layers
