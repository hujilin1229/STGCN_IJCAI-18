# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import z_score

import numpy as np
import pandas as pd
import os
import h5py

class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq

def data_gen(file_path, data_config, n_route, n_frame=21, day_slot=288):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset

def seq_gen_train_traffic4cast(data_seq, offset, n_frame, node_pos, C_0=3):
    '''
    Generate data in the form of standard sequence unit.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param num_nodes: int, the number of nodes in the graph.
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''

    num_data = data_seq.shape[0]
    n_slot = num_data - n_frame + 1
    tmp_seqs = []

    for i in range(0, n_slot, offset):
        tmp_seq = data_seq[i:i+n_frame, node_pos[:, 0], node_pos[:, 1], :C_0]
        tmp_seqs.append(tmp_seq)

    return np.stack(tmp_seqs, axis=0)


def data_gen_traffic4cast(file_path, process_dir, node_pos, seq_len, horizon, data_start,val_indices,
                          train_ratios=0.8, val_ratios=0.1):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''

    train_data_nz_file = process_dir + 'stgcn_seq{}_horizon{}_train_data.npz'.format(seq_len, horizon)
    val_data_nz_file = process_dir + 'stgcn_seq{}_horizon{}_train_data.npz'.format(seq_len, horizon)
    test_data_nz_file = process_dir + 'stgcn_seq{}_horizon{}_train_data.npz'.format(seq_len, horizon)

    if os.path.exists(train_data_nz_file):
        seq_train = np.load(train_data_nz_file)
        seq_train = seq_train['seq_data']

        seq_val = np.load(val_data_nz_file)
        seq_val = seq_val['seq_data']

        seq_test = np.load(test_data_nz_file)
        seq_test = seq_test['seq_data']
    else:
        files = os.listdir(file_path)
        data_list = []
        num_files = len(files)
        n_train = int(num_files * train_ratios)
        for f in files[:n_train]:
            try:
                data_file = h5py.File(file_path + '/' + f, 'r')
                raw_data = data_file['array'].value
                data_file.close()
                raw_data = raw_data[data_start:]
                tmp_data = seq_gen_train_traffic4cast(raw_data, horizon, seq_len+horizon, node_pos, C_0=3)
                data_list.append(tmp_data)
            except:
                print(file_path + '/' + f)
        seq_train = np.concatenate(data_list, axis=0)
        np.savez_compressed(train_data_nz_file, seq_data=seq_train)

        n_val = int(num_files * val_ratios)
        seq_val = []
        for f in files[n_train:n_train+n_val]:
            try:
                data_file = h5py.File(os.path.join(file_path, f), 'r')
                raw_data = data_file['array']
                data_file.close()
                seq_val += [raw_data[i-seq_len:i+horizon, node_pos[:, 0], node_pos[:, 1], :] for i in val_indices]
            except:
                print(file_path + '/' + f)
        seq_val = np.concatenate(seq_val, axis=0)
        np.savez_compressed(val_data_nz_file, seq_data=seq_val)

        seq_test = []
        for f in files[n_train + n_val:]:
            try:
                data_file = h5py.File(os.path.join(file_path, f), 'r')
                raw_data = data_file['array']
                data_file.close()
                seq_test += [raw_data[i - seq_len:i + horizon, node_pos[:, 0], node_pos[:, 1], :] for i in val_indices]
            except:
                print(file_path + '/' + f)
        seq_test = np.concatenate(seq_test, axis=0)
        np.savez_compressed(test_data_nz_file, seq_data=seq_test)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)

    return dataset

def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
