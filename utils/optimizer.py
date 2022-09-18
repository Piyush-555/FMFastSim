from enum import IntEnum
from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import Optimizer, Adadelta, Adagrad, Adam, Adamax, SGD, NAdam, RMSprop


class OptimizerType(IntEnum):
    """ Enum class of various optimizer types.

    This class must be IntEnum to be JSON serializable. This feature is important because, when Optuna's study is
    saved in a relational DB, all objects must be JSON serializable.
    """

    SGD = 0
    RMSPROP = 1
    ADAM = 2
    ADADELTA = 3
    ADAGRAD = 4
    ADAMAX = 5
    NADAM = 6
    # FTRL = 7  # Not in PyTorch


class OptimizerFactory:
    """Factory of optimizer like Stochastic Gradient Descent, RMSProp, Adam, etc.
    """

    @staticmethod
    def create_optimizer(parameters: Generator[Parameter, None, None],
        optimizer_type: OptimizerType, learning_rate: float, **kwargs) -> Optimizer:
        """For a given type and a learning rate creates an instance of optimizer.

        Args:
            optimizer_type: a type of optimizer
            learning_rate: a learning rate that should be passed to an optimizer

        Returns:
            An instance of optimizer.

        """
        if optimizer_type == OptimizerType.SGD:
            return SGD(parameters, learning_rate, **kwargs)
        elif optimizer_type == OptimizerType.RMSPROP:
            return RMSprop(parameters, learning_rate, **kwargs)
        elif optimizer_type == OptimizerType.ADAM:
            return Adam(parameters, learning_rate, **kwargs)
        elif optimizer_type == OptimizerType.ADADELTA:
            return Adadelta(parameters, learning_rate, **kwargs)
        elif optimizer_type == OptimizerType.ADAGRAD:
            return Adagrad(parameters, learning_rate, **kwargs)
        elif optimizer_type == OptimizerType.ADAMAX:
            return Adamax(parameters, learning_rate, **kwargs)
        elif optimizer_type == OptimizerType.NADAM:
            return Nadam(parameters, learning_rate, **kwargs)
        else:
            raise ValueError
            # Not in PyTorch
            """
            # i.e. optimizer_type == OptimizerType.FTRL
            return Ftrl(learning_rate)
            """
