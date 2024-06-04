# Reproduction of the paper entitled "Dimensionality-induced information loss of outliers in deep neural networks"

# Install libraries by PIP
```sh
pip install -r requirements.txt
```


# Training models (VGG and ResNet)

## Train ResNet-18 on CIFAR10 from scrach
```sh
python3 train_classifier.py --model-name resnet18 --traindata cifar10 --num_classes 10 --epochs 1200
```
Use --traindata and --model-name options to specify the training dataset and the model architecture, respectively.


# Evaluating models via propagation properties and internal features

## Singular values of weights, projection of features to weights, Gram matrices, noise sensitivity, etc
```sh
python3 analyzing.py --model-name resnet18 --traindata cifar10 --num_classes 10 --start_epoch 1200 --resume
```
You should run "analyzing.py" before running "kernels.py".

## Eigenvalues of covariances of features, Mahalanobis distance
```sh
python3 features.py --model-name resnet18 --traindata cifar10 --num_classes 10 --start_epoch 1200 --resume
```

## Eigenvalues of similarities between Gram matrices
```sh
python3 kernels.py --model-name resnet18 --traindata cifar10 --num_classes 10 --start_epoch 1200 --resume
```

## Converting to AUC, CKA, etc
```sh
python3 extractor.py --model-name resnet18 --traindata cifar10 --num_classes 10 --start_epoch 1200
```
This should be performed after running "analyzing.py", "features.py", "kernels.py".

## Visualizing statistic quantities

```sh
python3 averaging.py --num_seed 10
```
This should be performed after running "extractor.py".
It is recommended to construct a systematic directory structure like the following one.

/home/user/workdir/  
├── averaging.py  
├── cifar10/  
│ &nbsp; &nbsp; &nbsp; ├── vgg13/  
│ &nbsp; &nbsp; &nbsp; │ &nbsp; &nbsp; &nbsp; ├── rnd0/extract/  
│ &nbsp; &nbsp; &nbsp; │ &nbsp; &nbsp; &nbsp; └── rnd1/extract/  
│ &nbsp; &nbsp; &nbsp; └── resnet18/  
│ &emsp; &emsp; &emsp; &nbsp; ├── rnd0/extract/  
│ &emsp; &emsp; &emsp; &nbsp;  └── rnd1/extract/  
└── cifar100  
&nbsp; &nbsp; &nbsp; &nbsp; ├── vgg13/  
&nbsp; &nbsp; &nbsp; &nbsp; │   ├── rnd0/extract/  
&nbsp; &nbsp; &nbsp; &nbsp; │   └── rnd1/extract/  
&nbsp; &nbsp; &nbsp; &nbsp; └── resnet18/  
&emsp; &emsp; &emsp; &emsp; ├── rnd0/extract/  
&emsp; &emsp; &emsp; &emsp; └── rnd1/extract/  
