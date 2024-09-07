
# High-dimensional Anticounterfeiting Nanodiamonds Authenticated with Deep Metric Learning

## Environment Setup

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

## Data Preprocessing

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

## Training and Testing

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
