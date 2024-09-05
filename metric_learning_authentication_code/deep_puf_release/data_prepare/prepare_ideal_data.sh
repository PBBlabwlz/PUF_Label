# data_path="???"
# obtain current directory
current_dir=$(pwd)
py_path="${current_dir}/data_prepare/xls2npy.py"
save_path="./dataset/ideal_data"
source_train_txt="${current_dir}/data_prepare/ideal_train.txt"
source_test_txt="${current_dir}/data_prepare/ideal_test.txt"

mkdir ${save_path}
cd ${save_path}
python ${py_path} --split_make --total_num 300 --data_path ${data_path} --source_train_txt ${source_train_txt} --source_test_txt ${source_test_txt}
mkdir ./test_ref
mkdir ./test_sample
cp ./test_data/*_1.npy ./test_ref
cp ./test_data/*_2.npy ./test_sample
cd ..