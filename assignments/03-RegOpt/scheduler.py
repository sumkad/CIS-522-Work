from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom LR optimizer

    """

    lastStep: int

    def __init__(self, optimizer, last_epoch=-1):
        """
        Create a new scheduler.

        Arguments:
            optimizer: the optimizer used to modify the floats
            last_epoch: The last epoch of iteration

        """
        # ... Your Code Here ...
        self.lastStep = 0
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns the learning rate
        """
        return [i for i in self.base_lrs]

    def step(self, epoch=None) -> None:
        """
        Exponentially decreases the learning rate over time

        Arguments:
            epoch: the epoch of iteration
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)
        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        self.lastStep += 1
        if self.lastStep % 400 == 0:
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] *= 0.9
