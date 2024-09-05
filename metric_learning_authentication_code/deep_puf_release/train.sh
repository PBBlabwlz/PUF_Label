clean_data_path="./dataset/ideal_data"
noise_data_path="./dataset/noise_data" # only for evaluation

# train model on clean data and evaluate on noise data
python launch.py --config ./configs/base_dmp.yaml --train \
--gpu 1 trainer.num_nodes=1 \
data.scene_list=${clean_data_path} \
data.eval_scene_list=${noise_data_path}