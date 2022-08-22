# Spiking ResNet
***
Official implementation of the paper [Keys to Accurate Feature Extraction Using Residual Spiking Neural Networks](https://arxiv.org/abs/2111.05955)

Accepted in **IOP Neuromorphic Computing and Engineering**
### Installation
_Code tested in Python 3.8.10 and PyTorch 1.8.1_

Linux:\
`sh requirements`

Windows:\
`requirements.bat`

### Train S-ResNet
- CIFAR-10:
```
python train.py --num_steps 50 --lr 0.026844 --leak_mem 0.8744 --arch 'resnet_n' --dataset  'cifar10' --num_workers 4 --num_epochs 70
```
- CIFAR-100:
```
python train.py --num_steps 50 --lr 0.026844 --leak_mem 0.8744 --arch 'resnet_n' --dataset  'cifar100' --num_workers 4 --num_epochs 70
```
- DVS-CIFAR10:
```
    python train.py --num_steps 50 --lr 0.026844 --leak_mem 0.8744 --arch 'resnet_n_nm' --dataset  'cifar10dvs' --num_workers 4 --num_epochs 70
```
 ###### Other commands:
Set the batch size with the flag: `--batch_size` (use the biggest batch size your GPU can support)

You can set the GPU device to use with the flag `--device`

You can set S-ResNet's depth using the flag `--n` and its width using the flag `--nFilters`

To resume training from a saved checkpoint, indicate the checkpoint location using the flag `--reload`

If you want to fine tune a checkpoint trained with a different dataset, use the flag `--fine_tune`.
This will start the epoch count from 0, reset the accuracy history and skip the loading of the Fully
Connected layer and Conv1.

### Test S-ResNet
For testing, the location of the saved model is passed with `--model_path` alongside the architecture flags
such as `--arch` `--num_steps`. Remember to keep the same architecture and parameters that you used for training.

Also notice that the batch size can be bigger for inference than it was at training time.
- CIFAR-10:
```
python test.py --model_path "path to saved model" --num_steps 50 --arch 'sresnet' --dataset 'cifar10'
```
- CIFAR-100:
```
python test.py --model_path "path to saved model" --num_steps 50 --arch 'sresnet' --dataset 'cifar100'
```
- DVS-CIFAR10:
```
python test.py --model_path "path to saved model" --num_steps 50 --arch 'sresnet_nm' --dataset 'cifar10dvs'
```