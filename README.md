# Introduction
This contains data and code for our article 'High-dimensional Anticounterfeiting Nanodiamonds Authenticated with Deep Metric Learning'

Data: original readout images and digitized readout images of PUF labels acquired  in ideal lab condition and lab condition with additional noise sources.

Code: process and encode original readout images, authentication based on similarity index method, and authentication based on metric-learning method.

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
For our authentication algorithm, please refer to the folder `metric_learning_authentication_code/deep_puf_release` and its README.
