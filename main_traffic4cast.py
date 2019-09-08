# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test
import scipy.sparse as sp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--seq_len', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--city', type=str, default='Berlin', help='"Berlin", "Istanbul", "Moscow"')
parser.add_argument('--data_dir', type=str, default='../', help='directory to load the data')
parser.add_argument('--least_ratio', type=float, default=0.5, help='ratio to select pixels')
parser.add_argument('--inf_mode', type=str, default='merge')
args = parser.parse_args()
print(f'Training configs: {args}')

utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174,222, 258]

Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[3, 32, 64], [64, 32, 128]]

data_path = pjoin(args.data_dir, args.city)
process_data_dir = pjoin(data_path, 'process_{}'.format(args.least_ratio))
if not os.path.isdir(process_data_dir):
    os.mkdir(process_data_dir)
adj_file = pjoin(process_data_dir, 'adj_{}.npz'.format(args.least_ratio))
node_pos_file = pjoin(process_data_dir, 'node_pos_{}.npy'.format(args.least_ratio))

W = sp.load_npz(adj_file).toarray()
node_pos = np.load(node_pos_file)
# num of nodes
n = node_pos.shape[0]
args.n_route = n
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
Lk_sp = sp.coo_matrix(Lk)

# Lk_spt = tf.SparseTensorValue(
#     indices=np.array([Lk_sp.row, Lk_sp.col], np.int64).T,
#     values=Lk_sp.data,
#     dense_shape=Lk_sp.shape)

tf.add_to_collection(name='graph_kernel_indices', value=tf.cast(tf.constant(np.array([Lk_sp.row, Lk_sp.col]).T), tf.int64))
tf.add_to_collection(name='graph_kernel_value', value=tf.cast(tf.constant(Lk_sp.data), tf.float32))
tf.add_to_collection(name='graph_kernel_shape', value=tf.cast(tf.constant(Lk_sp.shape), tf.int64))

# tf.add_to_collection(name='graph_kernel', value=tf.cast(Lk_spt, tf.float32))

raw_data_path = pjoin(data_path, 'train_val') #the folder contain train and validation as the data used in the paper

indicies = utcPlus3
if args.city == 'Berlin':
    indicies = utcPlus2

traffic4cast_data = data_gen_traffic4cast(raw_data_path, process_data_dir, node_pos, args.seq_len, args.horizon,
                                          data_start=0, val_indices=indicies, train_ratios=0.8, val_ratios=0.1)
# Data Preprocessing
# data_file = f'PeMSD7_V_{n}.csv'
# n_train, n_val, n_test = 34, 5, 5
# PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {traffic4cast_data.mean:.2f}, STD: {traffic4cast_data.std:.2f}')

if __name__ == '__main__':
    model_train(traffic4cast_data, blocks, args, output_dim=3)
    model_test(traffic4cast_data, traffic4cast_data.get_len('test'), args.seq_len, args.horizon, args.inf_mode)
