from enum import Enum


class LearningRateScheduler(Enum):
    """Enum class for learning rate schedulers."""
    naive = "naive"
    const = "const"
    tunable = "tunable"

    def get_learning_rate(self, learning_rate: float, epoch: int, halve_after: int = 5) -> float:
        """
        Returns the learning rate scheduler.
        :param self: This learning rate scheduler
        :param learning_rate: Current learning rate
        :param epoch: Current epoch
        :param halve_after: Used for tunable learning rate scheduler. Halve the learning rate after this many epochs.
        :return: New learning rate
        """
        match self:
            case LearningRateScheduler.naive:
                return naive_learning_rate_scheduler(learning_rate, epoch)
            case LearningRateScheduler.const:
                return const_learning_rate_scheduler(learning_rate, epoch)
            case LearningRateScheduler.tunable:
                return tunable_learning_rate_scheduler(learning_rate, epoch, halve_after)


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


def tunable_learning_rate_scheduler(learning_rate: float, epoch: int, halve_after: int) -> float:
    """
    Tunable learning rate schedular that starts with learning_rate halves the learning rate every halving_parameter epochs.
    :param learning_rate: Current learning rate
    :param epoch: Current epoch
    :param halve_after: Halve the learning rate after this many epochs
    :return: New learning rate
    """
    if epoch < halve_after:
        if epoch == 0:
            return learning_rate
        return learning_rate + (learning_rate / epoch)
    if epoch % halve_after == 0 and epoch != 5 and epoch <= 20:
        return learning_rate * 0.5
    if epoch % 3 == 0 and epoch > 20:
        return learning_rate * 0.5
    return learning_rate
