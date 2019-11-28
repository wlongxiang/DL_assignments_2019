import sys

from torch import Tensor

from part1.dataset import PalindromeDataset
from part1.train import calc_accuracy


def test_palindrome_dataset_get_item():
    pd = PalindromeDataset(5)
    x, target = pd[0]
    print(x, target)
    assert x[0] == target


def test_palindrome_dataset_len():
    """
    by assigning sys.maxsize to __len__, we can force dataloader to train infinitely epochs.

    :return:
    """
    pd = PalindromeDataset(5)
    assert  len(pd) == sys.maxsize


def test_calc_accuracy_part1():
    # a 3-class problem
    targets = Tensor([0, 1, 1, 2])
    # predictions contain probability for each class index for each input, 4 input in this case, with 3 probs
    predictions = Tensor([
        [0.5, 0.1, 0.2],  # argmax at 0
        [0.3, 0.2, 0.11],  # argmax at 0
        [0.1, 1.0, 0.0],  # argmax at 1
        [0.00, 0.12, 0.9],  # argmax at 2

    ])
    # [0 ,1,1,2] == [0 , 0 , 1,2] will have three right prediction, 1 wrong prediction
    acc = calc_accuracy(predictions=predictions, targets=targets)
    print(acc)
    assert acc == 0.75
