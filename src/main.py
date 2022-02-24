import argparse

from experiment import Experiment


def parse_parameters():
    parser = argparse.ArgumentParser(description='This program is used to start one of the prepared experiments'
                                                 'concerning studying the importance of last layer of neural network'
                                                 'in the catastrophic forgetting during regularization based continual'
                                                 'class learning strategy.')
    parser.add_argument('--dataset', choices=['CIFAR100', 'MNIST', 'CUB200'], nargs='?', default='CIFAR100',
                        help='dataset on which strategy will be used')
    parser.add_argument('--epochs', type=int, help='number of epochs', default=80)
    parser.add_argument('--experiences', type=int, help='number of even class splits', default=5)
    parser.add_argument('--learning_rate', type=float, help='learning rate of Adam optimizer', default=0.001)
    parser.add_argument('--batch_size', type=int,
                        help='the train minibatch size', default=32)

    subparsers = parser.add_subparsers(help='specific continual learning algorithm type', dest='strategy',
                                       required=True, metavar='algorithm')

    # EWC
    parser_ewc = subparsers.add_parser('EWC', help='Elastic Weight Consolidation')
    parser_ewc.add_argument('--ewc_lambda', type=float,
                        help='EWC hyperparameter to weigh the penalty inside the total loss. The larger the lambda, '
                             'the larger the regularization',
                        default=0.1)

    parser_ewc.add_argument('--ewc_mode', choices=['separate, onlinesum'], nargs='?', default='separate',
                        help='separate to keep a separate penalty for each previous experience. onlinesum to keep a '
                             'single penalty summed over all previous tasks. onlineweightedsum to keep a single '
                             'penalty summed with a decay factor over all previous tasks')
    parser_ewc.add_argument('--ewc_decay_factor', type=float, default=None,
                        help='used only if mode is onlineweightedsum. It specify the decay term of the importance '
                             'matrix')
    parser_ewc.add_argument('--ewc_keep_importance_data', type=bool, default=True,
                        help='if True, keep in memory both parameter values and importances for all previous task, '
                             'for all modes. If False, keep only last parameter values and importances. If mode is '
                             'separate, the value of keep_importance_data is set to be True.')

    # LWF
    parser_lwf = subparsers.add_parser('LWF', help='Learning without Forgetting')

    parser_lwf.add_argument('--lwf_alpha', type=float,
                        help='LwF distillation hyperparameter. It can be either a float number or a list containing '
                             'alpha for each experience.',
                        default=0.2)
    parser_lwf.add_argument('--lwf_temperature', type=float, help='LwF softmax temperature for distillation', default=3)

    # SI
    parser_si = subparsers.add_parser('SI', help='Synaptic Intelligence')

    parser_si.add_argument('--si_lambda', type=float,
                        help='Synaptic Intelligence lambda term. If list, one lambda for each experience. If the list '
                             'has less elements than the number of experiences, last lambda will be used for the '
                             'remaining experiences', default=3)
    parser_si.add_argument('--si_eps', type=float,
                        help='Synaptic Intelligence damping parameter.', default=1e-07)

    # JOINT
    parser_ewc = subparsers.add_parser('JOINT', help='Joint training')


    parser.print_help()
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_parameters()
    experiment = Experiment(args)
    experiment.run_experiment()

if __name__ == '__main__':
    main()
