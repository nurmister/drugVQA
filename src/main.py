"""Add docstring here."""

import torch
import warnings
from trainAndTest import train, trainArgs

#  from sklearn import metrics

warnings.filterwarnings("ignore")
torch.cuda.set_device(0)


def main():
    """Parse command line parameters, reading data, fitting and scoring a SEAL-CI model."""
    losses, accs, testResults = train(trainArgs)


if __name__ == "__main__":
    main()
