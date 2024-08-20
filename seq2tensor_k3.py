import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from sklearn import preprocessing
from transformers import BertModel, BertTokenizer
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#将DNA序列转为k-mers
def build_kmers(seq, k_size):
    """a function to calculate kmers from seq"""
    kmers = []  # k-mer存储在列表中
    n_kmers = len(seq) - k_size + 1
    for i in range(n_kmers):
        kmer = seq[i:i + k_size]
        kmers.append(kmer)
    return kmers

#DNA序列转为k-mer，将所有的k-mer转为向量值，针对单条序列（针对DNABERT模型设置）
def seq_list_k_mer_num(k_mers_list, dnabert_m):
    # 加载序列表示模型
    MODELNAME = dnabert_m  # 'bert_gg_k_mer_3_108/k_3-108_dnabert'
    tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
    model = BertModel.from_pretrained(MODELNAME)  # 模型
    model.eval()
    model.cuda()
    kmers_num = []  #k-mer的向量值存储在列表中
    for g in range(0, len(k_mers_list), 800):
        k_mers_g = k_mers_list[g:g + 800]

        input_ids = torch.tensor(tokenizer.encode(k_mers_g)).unsqueeze(0).cuda()
        outputs = model(input_ids)
        sequence_output = outputs[0][0][1:int((len(k_mers_g) + 1)/(3 + 1)) + 1].tolist()

        kmers_num.extend(sequence_output)
    tt = len(kmers_num)
    kmers_num = list(map(list, zip(*kmers_num)))
    #print(len(kmers_num[0]))

    m = 4000 - tt

    kmers_num = np.pad(kmers_num, ((0, 0), (0, m)), 'constant', constant_values=0)

    return kmers_num

def seqs_list_k_mer_num(many_k_mers_list, dnabert_m):
    seq_num = []
    q = 1
    for seq_ll in many_k_mers_list:
        #print('第q条序列开始处理：q=', q)
        seq_num.append(seq_list_k_mer_num(seq_ll, dnabert_m))
        q = q + 1
    return seq_num
def seq2tensor_k3(fa_path, seqchunk, num_path, dnabert_m):

    bac_seqs = [str(fa.seq) for fa in SeqIO.parse(fa_path,  "fasta")]

    seq_kmers = []
    for i in bac_seqs:
        seq_kmer = build_kmers(i, 3)
        seq_kmer = ' '.join(seq_kmer)
        seq_kmers.append(seq_kmer)

    #第一步：将分割为k-mers序列的DNA序列进行向量赋值
    for i in range(0, len(seq_kmers), seqchunk):
        #print(len(seq_kmers[i:i+2]))

        #k_mer_matrix_train = seqs_list_k_mer_num(seq_kmers[i:i+2], dnabert_m)
        k_mer_matrix_train = seqs_list_k_mer_num(seq_kmers, dnabert_m)
        k_mer_matrix_train = torch.tensor(k_mer_matrix_train, dtype=torch.float32)
        k_mer_matrix_train = k_mer_matrix_train.unsqueeze(1)

        path_str = num_path + str(i) +'.pth'
        #torch.save(x_train, 'ss_train_k_mer3.pth')
        torch.save(k_mer_matrix_train, path_str)

def main():
    #1 使用的第一步argparse是创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='manual to this script')
    #2add_argument()添加参数。
    parser.add_argument("--i_dnabert_m", type=str, help="The dnabert model file path. ", default='k_3-108_dnabert', required=False)
    parser.add_argument("--i_fasta", type=str, help="The fasta file path to be annotated. ", required=True)
    parser.add_argument("--i_seqchunk", type=str, help="How many sequences are processed each time.", default=10000, required=False)
    parser.add_argument("--o_seq2num", type=str, help="Processed file path without file suffix. ", required=False)
    #3.解析参数
    args = parser.parse_args()
    seq2tensor_k3(args.i_fasta, args.i_seqchunk, args.o_seq2num, args.i_dnabert_m)

if __name__ == "__main__":
    main()
