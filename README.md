# regularization-algorithms-last-layer-significance-experiment

This program is used to start one of the prepared experimentsconcerning studying the importance of last layer of neural networkin the catastrophic forgetting during regularization based continualclass learning strategy.

Project works on Linux os.

## How to run:

1. Install requirements `pip install -r requirements.txt`
2. Run `./run-experiments.sh` or start specific experiment manually:

This program is used to start one of the prepared experimentsconcerning studying the importance of last layer of neural networkin the catastrophic forgetting during regularization based continualclass learning strategy.

positional arguments:
  algorithm             specific continual learning algorithm type
    JOINT               Joint training
    EWC                 Elastic Weight Consolidation
    LWF                 Learning without Forgetting
    SI                  Synaptic Intelligence
    ICARL               Incremental Classifier and Representation Learning
    NAIVE               Naive finetuning

options:
  -h, --help            show this help message and exit
  --dataset [{CIFAR100,MNIST,CUB200}]
                        dataset on which strategy will be used
  --epochs EPOCHS       number of epochs for training using main algorithm
  --last_layer_epochs LAST_LAYER_EPOCHS
                        number of epochs for finetuning last layer
  --experiences EXPERIENCES
                        number of even class splits
  --learning_rate LEARNING_RATE
                        learning rate of Adam optimizer
  --batch_size BATCH_SIZE
                        the train minibatch size
  --cpu                 If CPU should be used instead of CUDA
