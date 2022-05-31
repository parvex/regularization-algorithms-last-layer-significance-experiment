python src/main.py --dataset MNIST --epochs 5  --experiences 5  --learning_rate 1e-3 --batch_size 128 JOINT
python src/main.py --dataset MNIST --epochs 5  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 2 NAIVE
python src/main.py --dataset MNIST --epochs 5  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 2 ICARL --icarl_memory_size 2000
python src/main.py --dataset MNIST --epochs 5  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 2 LWF --lwf_alpha 1 --lwf_temperature 2
python src/main.py --dataset MNIST --epochs 5  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 2 EWC --ewc_lambda 2 --ewc_mode separate
python src/main.py --dataset MNIST --epochs 5  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 2 SI --si_lambda 0.1 --si_eps 1e-03

python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 JOINT
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 NAIVE
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 ICARL --icarl_memory_size 2000
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 LWF --lwf_alpha 1 --lwf_temperature 2
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 EWC --ewc_lambda 2 --ewc_mode separate
python src/main.py --dataset CIFAR100 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 SI --si_lambda 0.1 --si_eps 1e-03

#python src/main.py --dataset CUB200 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 JOINT
#python src/main.py --dataset CUB200 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 NAIVE
#python src/main.py --dataset CUB200 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 ICARL --icarl_memory_size 2000
#python src/main.py --dataset CUB200 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 LWF --lwf_alpha 1 --lwf_temperature 2
#python src/main.py --dataset CUB200 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 EWC --ewc_lambda 2 --ewc_mode separate
#python src/main.py --dataset CUB200 --epochs 70  --experiences 5  --learning_rate 1e-3 --batch_size 128 --last_layer_epochs 20 SI --si_lambda 0.1 --si_eps 1e-03