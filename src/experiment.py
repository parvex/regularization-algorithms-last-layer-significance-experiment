import copy
import os

import randomname
import torch
from avalanche.benchmarks import SplitCIFAR100, SplitMNIST, SplitCUB200
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger
from avalanche.models import make_icarl_net
from avalanche.training import EWC, LwF, SynapticIntelligence, JointTraining, ICaRL, Naive
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class Experiment:
    device = None
    args = None
    benchmark = None
    model = None
    eval_plugin = None
    optimizer = None
    scheduler = None
    strategy = None
    plugins = None

    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        print('device: ' + str(self.device))

        self.args = args
        self.benchmark = self.get_benchmark(args)
        n_channels = 1 if args.dataset == 'MNIST' else 3
        self.model = make_icarl_net(num_classes=self.benchmark.n_classes, n=5, c=n_channels)

        self.eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            benchmark=self.benchmark, loggers=[
                InteractiveLogger(),
                TextLogger(open('log.txt', 'a')),
                WandBLogger(project_name="reg-alg-cl-last-layer-importance",
                            run_name=f"Mgr[{args.dataset}, {args.strategy}-{randomname.get_name()}]")]
        )

        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)

        # todo if needed
        self.scheduler = None
        # todo if needed
        self.plugins = None

        self.strategy = self.get_strategy(args)

    # TRAINING LOOP
    def run_experiment(self):
        print('Starting experiment...')
        results = []

        if self.args.strategy == 'JOINT':
            print('JOINT TRAINING - UPPER BOUND')
            self.strategy.train(self.benchmark.train_stream)
            print('EVAL ON JOINT TRAINING')
            results.append(self.strategy.eval(self.benchmark.test_stream))
            return

        for experience in self.benchmark.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            # train returns a dictionary which contains all the metric values
            res = self.strategy.train(experience)
            print('Training completed')

            print('Computing accuracy on the whole test set')
            # test also returns a dictionary which contains all the metric values
            results.append(self.strategy.eval(self.benchmark.test_stream))

        frozen_model = copy.deepcopy(self.model)
        frozen_model.feature_extractor.requires_grad_(False)
        strategy = JointTraining(self.model, self.optimizer, CrossEntropyLoss(),
                                 train_mb_size=self.args.batch_size, eval_mb_size=self.args.batch_size,
                                 train_epochs=self.args.last_layer_epochs, evaluator=self.eval_plugin, device=self.device,
                                 plugins=self.plugins)

        print('JOINT TRAINING ONLY ON LAST LAYER')
        strategy.train(self.benchmark.train_stream)

        print('EVAL ON JOINT TRAINING ONLY ON LAST LAYER')
        results.append(strategy.eval(self.benchmark.test_stream))

    def get_benchmark(self, args):
        if args.dataset == 'CIFAR100':
            return SplitCIFAR100(n_experiences=args.experiences)
        elif args.dataset == 'MNIST':
            return SplitMNIST(n_experiences=args.experiences)
        elif args.dataset == 'CUB200':
            return SplitCUB200(n_experiences=args.experiences)

    def get_strategy(self, args):
        if args.strategy == 'JOINT':
            return JointTraining(self.model, self.optimizer, CrossEntropyLoss(),
                                 train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                 train_epochs=args.epochs, evaluator=self.eval_plugin, device=self.device,
                                 plugins=self.plugins)
        if args.strategy == 'NAIVE':
            return Naive(self.model, self.optimizer, CrossEntropyLoss(),
                         train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                         train_epochs=args.epochs, evaluator=self.eval_plugin, device=self.device,
                         plugins=self.plugins)
        elif args.strategy == 'EWC':
            return EWC(self.model, self.optimizer, CrossEntropyLoss(),
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       train_epochs=args.epochs, evaluator=self.eval_plugin, device=self.device, plugins=self.plugins,
                       ewc_lambda=args.ewc_lambda, mode=args.ewc_mode, decay_factor=args.ewc_decay_factor,
                       keep_importance_data=args.ewc_keep_importance_data)
        elif args.strategy == 'LWF':
            return LwF(self.model, self.optimizer, CrossEntropyLoss(),
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       train_epochs=args.epochs, evaluator=self.eval_plugin, device=self.device, plugins=self.plugins,
                       alpha=args.lwf_alpha, temperature=args.lwf_temperature)
        elif args.strategy == 'SI':
            return SynapticIntelligence(self.model, self.optimizer, CrossEntropyLoss(),
                                        train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                        train_epochs=args.epochs, evaluator=self.eval_plugin, device=self.device,
                                        plugins=self.plugins,
                                        si_lambda=args.si_lambda, eps=args.si_eps)
        elif args.strategy == 'ICARL':
            return ICaRL(self.model.feature_extractor, self.model.classifier, self.optimizer,
                         train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                         train_epochs=args.epochs, evaluator=self.eval_plugin, device=self.device,
                         plugins=self.plugins,
                         memory_size=args.icarl_memory_size, fixed_memory=True, buffer_transform=None)
