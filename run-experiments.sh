python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 JOINT
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 NAIVE
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 EWC --ewc_lambda 0.2 --ewc_mode separate
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 LWF --lwf_alpha 0.2 --lwf_temperature 3
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 SI --si_lambda 3 --si_eps 1e-07
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 ICARL --icarl_memory_size 2000