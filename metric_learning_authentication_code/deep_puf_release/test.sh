clean_data_path="./dataset/ideal_data"
noise_data_path="./dataset/noise_data" # only for evaluation
ckpt_path="./outputs/base_dmp/scratch@20240905-133223/ckpts/last.ckpt" # e.g., ./outputs/base_dmp/scratch@20240905-133223/ckpts/last.ckpt

# evalaute the model on clean data
python launch.py --config ./configs/base_dmp.yaml --test \
--gpu 1 trainer.num_nodes=1 \
resume=${ckpt_path} \
data.eval_scene_list=${clean_data_path} tag="clean_test"

# evalaute the model on noise data
python launch.py --config ./configs/base_dmp.yaml --test \
--gpu 1 trainer.num_nodes=1 \
resume=${ckpt_path} \
data.eval_scene_list=${noise_data_path} tag="noise_test"