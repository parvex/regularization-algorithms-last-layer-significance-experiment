import copy
import sys

import torch
import uuid
import wandb
from avalanche.benchmarks import SplitCIFAR100
from avalanche.training import JointTraining, Replay, ICaRL
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP, SimpleCNN, make_icarl_net
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import EWC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benchmark = SplitCIFAR100(n_experiences=5)
model = make_icarl_net(num_classes=benchmark.n_classes)

if len(sys.argv) > 1:
    wandb.login(key=sys.argv[1])
else:
    print('wandb api key was not specified as arg1, logging manually')

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    benchmark=benchmark, loggers=[
        InteractiveLogger(),
        TextLogger(open('log.txt', 'a')),
        TensorboardLogger(),
        WandBLogger(project_name="reg-alg-cl-last-layer-importance", run_name=f"test-icarl-{uuid.uuid4()}")]
)

strategy = ICaRL(
    model.feature_extractor, model.classifier, SGD(model.parameters(), lr=0.001, momentum=0.9),
    buffer_transform=None, memory_size=20, fixed_memory=False,
    train_mb_size=128, train_epochs=60, eval_mb_size=128,
    evaluator=eval_plugin, device=device)


# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(strategy.eval(benchmark.test_stream))

#
# freeze_model = copy.deepcopy(model)
# freeze_model.features.requires_grad_(False)
# strategy = JointTraining(
#     freeze_model, SGD(freeze_model.parameters(), lr=0.001, momentum=0.9),
#     CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
#     evaluator=eval_plugin)
# print('JOINT TRAINING ONLY ON LAST LAYER')
# strategy.train(benchmark.train_stream)
#
# print('EVAL ON JOINT TRAINING ONLY ON LAST LAYER')
# results.append(strategy.eval(benchmark.test_stream))
