python src/main.py --dataset CIFAR100 --epochs 80  --experiences 5  --learning_rate 1e-3 --batch_size 32 JOINT
python src/main.py --dataset CIFAR100 --epochs 80  --experiences 5  --learning_rate 1e-3 --batch_size 32 EWC --ewc_lambda 0.2 --ewc_mode separate
python src/main.py --dataset CIFAR100 --epochs 80  --experiences 5  --learning_rate 1e-3 --batch_size 32 LWF --lwf_alpha 0.2 --lwf_temperature 3
python src/main.py --dataset CIFAR100 --epochs 80  --experiences 5  --learning_rate 1e-3 --batch_size 32 SI --si_lambda 3 --si_eps 1e-07