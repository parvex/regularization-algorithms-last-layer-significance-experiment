import randomname
import torch
from avalanche.benchmarks import SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger
from avalanche.models import make_icarl_net
from avalanche.training import ICaRL
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device {device}')

benchmark = SplitCIFAR100(n_experiences=5)
model = make_icarl_net(num_classes=benchmark.n_classes, n=5)

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    benchmark=benchmark, loggers=[
        InteractiveLogger(),
        TextLogger(open('log.txt', 'a')),
        WandBLogger(project_name="reg-alg-cl-last-layer-importance", run_name=f"test-icarl:[{randomname.get_name()}]")]
)

optimizer = SGD(model.parameters(), lr=2, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[49, 63], gamma=0.2)

strategy = ICaRL(
    model.feature_extractor, model.classifier, optimizer,
    buffer_transform=None, memory_size=2000, fixed_memory=True,
    train_mb_size=128, train_epochs=70, eval_mb_size=128,
    evaluator=eval_plugin, device=device,
    plugins=[LRSchedulerPlugin(scheduler)]
)


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
