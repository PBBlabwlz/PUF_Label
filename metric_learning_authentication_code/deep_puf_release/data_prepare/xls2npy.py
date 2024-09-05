import numpy as np
import pandas as pd
import os
from shutil import copyfile
import random
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--split_make', action='store_true',
                       help='if true, split data into train and test, this will genrate train.txt and test.txt')
argparser.add_argument('--train_percentage', type=float, default=0.8,
                       help='train percentage, default is 0.8')
argparser.add_argument('--total_num', type=int, default=100,
                       help='total number of data, default is 100')
argparser.add_argument('--data_path', type=str, default='./data',
                       help='data path, default is ./data')
argparser.add_argument('--source_train_txt', type=str, default=None,
                          help='source train txt, default is None')
argparser.add_argument('--source_test_txt', type=str, default=None,
                            help='source test txt, default is None')
argparser.add_argument('--source_train_nums', type=int, default=None,
                            help='source train nums, default is None')
argparser.add_argument('--source_test_nums', type=int, default=None,    
                            help='source test nums, default is None')

opt = argparser.parse_args()

NUM = opt.total_num
ANGLES = [36, 48, 60, 72, 84, 96, 108, 120, 132, 144]

def split_list(
    num=NUM, 
    percentage=0.8, 
    source_train_txt=None, 
    source_test_txt=None,
    source_train_nums=None,
    source_test_nums=None,
    ):
    if source_train_txt!=None and source_test_txt!=None:
        with open(source_train_txt, 'r') as f:
            train_idx = [line.strip('\n') for line in f]
            if source_train_nums!=None:
                train_idx = train_idx[:source_train_nums]
        with open(source_test_txt, 'r') as f:
            test_idx = [line.strip('\n') for line in f]
            if source_test_nums!=None:
                test_idx = test_idx[:source_test_nums]
        with open("train.txt", 'w') as f:
            for i in train_idx:
                f.write(str(i) + '\n')
        with open("test.txt", 'w') as f:
            for i in test_idx:
                f.write(str(i) + '\n')
    else:
        li = [x for x in range(0, num)]
        random.shuffle(li)
        per = int(percentage * num)
        train_idx = li[:per]
        test_idx = li[per:]
        train_idx.sort()
        test_idx.sort()
        with open("train.txt", 'w') as f:
            for i in train_idx:
                f.write(str(i) + '\n')
        with open("test.txt", 'w') as f:
            for i in test_idx:
                f.write(str(i) + '\n')

def make_npy(path):
    g = os.walk(path)
    outdir = 'full_data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for dirpaths, dirnames, filenames in g:
        for dir in dirnames:
            sample_dir = os.path.join(path, dir)
            # if sample_dir contains no files, skip
            if not os.listdir(sample_dir):
                continue
            out_path = os.path.join(outdir, '%s.npy' %dir)
            my_list = []
            for angle in ANGLES:
                # sample_channel = os.path.join(sample_dir, '%02d.xlsx' %angle)
                sample_channel = os.path.join(sample_dir, '%d.xlsx' %angle)
                print(sample_channel)
                data = pd.read_excel(sample_channel, header=None)
                my_list.append(data)
            npdata = np.array(my_list)
            np.save(out_path, npdata)

def split_train_test(num=NUM):
    train_path = 'train_data'
    test_path = 'test_data'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    path = './full_data'

    train_idx = []
    with open('train.txt', 'r') as f:
        for line in f:
            train_idx.append(line.strip('\n'))
    print(train_idx)

    test_idx = []
    with open('test.txt', 'r') as f:
        for line in f:
            test_idx.append(line.strip('\n'))

    for index in range(0, num):
        name_1 = '%03d' % index + '_1.npy'
        name_2 = '%03d' % index + '_2.npy'
        src_1 = os.path.join(path, name_1)
        src_2 = os.path.join(path, name_2)

        if str(index) in train_idx:
            dst_1 = os.path.join(train_path, name_1)
            dst_2 = os.path.join(train_path, name_2)
            copyfile(src_1, dst_1)
            copyfile(src_2, dst_2)
        else:
            dst_1 = os.path.join(test_path, name_1)
            dst_2 = os.path.join(test_path, name_2)
            copyfile(src_1, dst_1)
            copyfile(src_2, dst_2)

def test_fuse():
    test_fuse_path = 'test_fuse'
    test_path = './test_data'
    if not os.path.exists(test_fuse_path):
        os.mkdir(test_fuse_path)
    test_input_list = sorted([file for file in os.listdir(test_path) if file.endswith('.npy')])
    n = len(test_input_list)
    for i in range(n):
        file_name_1 = test_input_list[i]
        name_1 = file_name_1.split(os.sep)[-1].split('.')[0]
        data_1 = np.load(os.path.join(test_path, file_name_1))
        for j in range(i+1, n):
            file_name_2 = test_input_list[j]
            name_2 = file_name_2.split(os.sep)[-1].split('.')[0]
            data_2 = np.load(os.path.join(test_path, file_name_2))

            data = np.concatenate((data_1, data_2), axis=0)
            name = name_1+'_'+name_2
            out_path = os.path.join(test_fuse_path, name+'.npy')

            np.save(out_path, data)


if __name__ == '__main__':
    if opt.split_make:
        if opt.source_train_txt!=None and opt.source_test_txt!=None:
            print('split data into train and test list from source txt...')
            split_list(
                source_train_txt=opt.source_train_txt, 
                source_test_txt=opt.source_test_txt,
                source_train_nums=opt.source_train_nums,
                source_test_nums=opt.source_test_nums,
            )
        else:
            assert opt.source_train_txt == None and opt.source_test_txt == None
            print('split data into train and test list...')
            split_list(percentage=opt.train_percentage)
    print('make npy...')
    data_path = opt.data_path
    make_npy(data_path)
    print('split train and test data...')
    split_train_test()
    # print('test fuse...')
    # test_fuse()
    print('done!')