# 3-Body with Neural Network

This is a proposed algorithm to solve 3-body problem using neural network used for the assignment in the course Parallel Computing. In this repo, we've already provided **SIMPLE and EASY-to-handle command line tools, a pretrained model and preprocessed dataset**.

# Minimum Requirements  
Disk Size: 35MB  
RAM: 16GB  
GPU Memory Capacity: 4 GB  
CPU Memory Capacity: 16 GB  
Operation System: Window 10 or Ubuntu 20.04 LTS

Note: ***If you use Linux, replace python as python3 when using the command line***

# Dependency
python==3.7.4  
torch==1.7.1  
torchvision==0.8.2  
numpy==1.19.2  
cupy-cuda101=8.1.0 

# How to use

## Data Preparation (Optional)

1. Generate the data:  
``cd ANN``  
``python generate_data.py --data_size dataSize``

## Artificial Neural Network

### Training (Optional)

1. Once obtain the dataset, go to the **ANN** folder:
``cd ANN``
2. Train the model:
``python 3-body_ann.py --num_epochs num_epoch --saved_every_epoch period_to_save --batch_size batch_size``

### Inference

#### Test accuracy of the model

1. Go to the **ANN** folder:
``cd ANN``
2. Use following command:
``python 3-body_ann_test_accuracy.py --model model_path``

#### Test speed of the model

1. Go to the **ANN** folder:
``cd ANN``
2. Use following command:
``python 3-body_ann_test_speed.py --model model_path --end_time endTime``

## Normal Parallel Implementation

### Test speed of NumPy implementation
``python 3-body_numpy.py --end_time endTime``

### Test speed of CuPy implementation
``python 3-body_cupy.py --end_time endTime``