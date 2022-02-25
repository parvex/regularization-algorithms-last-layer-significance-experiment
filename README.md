# regularization-algorithms-last-layer-significance-experiment

This program is used to start one of the prepared experimentsconcerning studying the importance of last layer of neural networkin the catastrophic forgetting during regularization based continualclass learning strategy.

Project works on Linux os.

## How to run:

1. Install Avalanche `pip install avalanche-lib`
2. Run `./run-experiments.sh` or start specific experiment manually:

usage: main.py [-h] [--dataset [{CIFAR100,MNIST,CUB200}]] [--epochs EPOCHS] [--experiences EXPERIENCES] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] algorithm ...

positional arguments:
  algorithm             specific continual learning algorithm type
    EWC                 Elastic Weight Consolidation
    LWF                 Learning without Forgetting
    SI                  Synaptic Intelligence
    JOINT               Joint training

optional arguments:
  -h, --help            show this help message and exit
  --dataset [{CIFAR100,MNIST,CUB200}]
                        dataset on which strategy will be used
  --epochs EPOCHS       number of epochs
  --experiences EXPERIENCES
                        number of even class splits
  --learning_rate LEARNING_RATE
                        learning rate of Adam optimizer
  --batch_size BATCH_SIZE
                        the train minibatch size
usage: main.py [-h] [--dataset [{CIFAR100,MNIST,CUB200}]] [--epochs EPOCHS] [--experiences EXPERIENCES] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] algorithm ...

