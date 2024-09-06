# Introduction
This contains data and code for our article 'High-dimensional Anticounterfeiting Nanodiamonds Authenticated with Deep Metric Learning'

Data: original readout images and digitized readout images of PUF labels acquired  in ideal lab condition and lab condition with additional noise sources.

Code: process and encode original readout images, authentication based on similarity index method, and XXX.

# Dataset
Dataset is contained in following folders.

`ideal_data`: two times of readout results of 300 PUF labels acquired in ideal lab condition

`noise_data`: two times of readout results of 150 PUF labels acquired in lab condition with additional noise source

`ideal_digitized_images`: digitized images corresponding to the readout results of 300 PUF labels acquired in ideal lab condition

`noise_digitized_images`: digitized images corresponding to the readout results of 150 PUF labels acquired in lab condition with additional noise source

# Encode and Similarity Index Authentication code
MATLAB code runs in MATLAB R2023b.
## 1) Encode Original Images
`encode_images.m` in 'encoding code' folder: this file process and encode the original readout images, and save the digitized images.

`bandpass.m` in 'encoding code' folder: please put this file and `encode_images.m` in the same folder, when running `encode_images.m`.


Please input following parameters in `encode_images.m` before running:

```bash
num_label=300; % total number of PUF label, 300 means 300 PUF labels
index_data="_1"; % readout time of PUF label, 1 means the first time of readout results
index_folder_save="_1\"; % readout time of PUF label, 1 means the first time of readout results
open_name1='D:\test\ideal_data\'; % path of folder saving original readout results
save_name1='D:\test\ideal_digitized_images\'; % path of folder saving digitized readout results
```

Please remember to create folders with name shown in 'ideal_digitized_images' folder to save the digitized readout results.

## 2) Authentication based on similarity index method
`similarity_index.m` in 'similarity_index_authentication_code' folder: this file authenticate the digitized images via similarity index method, and authentication result is saved in variable called sim_index.


Please input following parameters in `similarity_index.m` before running:

```bash
num_label=300; % total number of PUF labels, 300 means 300 PUF labels
file_name1='D:\test\ideal_digitized_images\'; % path of folder saving digitized readout results
```

# Metirc Learning Authentication
## 1) Environment Setup

This project requires PyTorch and additional packages listed in `requirements.txt`.

First, create a new Python environment and activate it with python 3.8:

```bash
conda create -n deep_puf python=3.8
conda activate deep_puf
```

To install PyTorch, please follow the instructions from the official [PyTorch website](https://pytorch.org/get-started/locally/) to choose the appropriate version for your system. For our experiments, we used PyTorch 2.3.1 with CUDA 11.8. In this case, you can simply run:
    
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

To install the necessary Python packages, run the following command:

```bash
pip install -r requirements.txt
```

## 2) Data Preprocessing

Before training, the raw data needs to be preprocessed to generate the training and testing datasets. Please ensure that the processed data is placed in the `dataset` folder. 

Preprocessing can be done by running the following script:

```bash
export data_path="path/to/your/data"
bash data_prepare/prepare_ideal_data.sh 
```

where `data_path` is the path to the raw clean data you downloaded.
This will automatically process the data under ideal lab condition, which will be used for training and test.

To process the data under noise-perturbed conditions, run the following script:

```bash
export data_path="path/to/your/data"
bash data_prepare/prepare_noise_data.sh 
```

where `data_path` is the path to the raw noise data you downloaded.
Note that this will be only used for test! The train_data under the processed folder would be empty.

## 3) Training and Testing

### Training

To train the model, use the following command:

```bash
bash train.sh
```

This will start the training process using the clean data, and automatically evaluate the test results on noise data when the training completes.

### Testing

Once training is completed, you can test the model with the following command:

```bash
bash test.sh
```

where you need to specify the ckpt path in the `test.sh` file.
This will evaluate the model performance on the test dataset under both ideal dataset and noise dataset.

All the results can be found in the `outputs` folder.
