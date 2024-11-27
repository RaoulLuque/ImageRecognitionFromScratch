from enum import Enum


class LearningRateScheduler(Enum):
    """Enum class for learning rate schedulers."""
    naive = "naive"
    const = "const"
    tuned = "tuned"

    def get_learning_rate(self, learning_rate: float, epoch: int) -> float:
        """
        Returns the learning rate scheduler.
        :param self: This learning rate scheduler
        :param learning_rate: Current learning rate
        :param epoch: Current epoch
        :return: New learning rate
        """
        match self:
            case LearningRateScheduler.naive:
                return naive_learning_rate_scheduler(learning_rate, epoch)
            case LearningRateScheduler.const:
                return const_learning_rate_scheduler(learning_rate, epoch)
            case LearningRateScheduler.tuned:
                return tuned_learning_rate_scheduler(learning_rate, epoch)


def naive_learning_rate_scheduler(learning_rate: float, epoch: int) -> float:
    """
    Naive learning rate schedular that starts with 0.0005 halves the learning rate every 5 epochs.
    :param learning_rate: Current learning rate
    :param epoch: Current epoch
    :return: New learning rate
    """
    initial_lr = 0.0005
    if epoch < 5:
        return initial_lr * (epoch + 1) / 5
    return initial_lr * 0.5 ** ((epoch - 5) // 5)


def const_learning_rate_scheduler(learning_rate: float, epoch: int) -> float:
    """
    Returns the input learning rate.
    :param learning_rate: Learning rate that is returned
    :param epoch: Ignored
    :return: Returns the input learning rate
    """
    return learning_rate


def tuned_learning_rate_scheduler(learning_rate: float, epoch: int) -> float:
    """
    Naive learning rate schedular that starts with 0.0005 halves the learning rate every 5 epochs.
    :param learning_rate: Current learning rate
    :param epoch: Current epoch
    :return: New learning rate
    """
    starting_lr = 0.005
    if epoch < 10:
        return starting_lr * (epoch + 1) / 10
    later_lr = 0.0005
    return later_lr * (0.5 ** ((epoch - 10) // 5))
