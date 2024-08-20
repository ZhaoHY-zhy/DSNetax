import argparse

from seq2tensor_k3 import seq2tensor_k3
from DSNetax_result import DSNetax_classifation

def main():
    #1 使用的第一步argparse是创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='manual to this script')
    #2add_argument()添加参数。
    parser.add_argument("--i_dnabert_m", type=str, help="The dnabert model file path. ", default='k_3-108_dnabert', required=False)
    parser.add_argument("--i_DSNetax_m", type=str, help="The DSNetax model file path. ", default='trained_DSNetax.pth', required=False)
    parser.add_argument("--i_result_nums", type=str, help="Number of output results. ", default=1, choices=['1', '5'], required=False)
    parser.add_argument("--i_seqchunk", type=str, help="How many sequences are processed each time.", default=1000, required=False)

    parser.add_argument("--i_fasta", type=str, help="The fasta file path to be annotated. ", required=True)
    parser.add_argument("--o_seq2num", type=str, help="Processed file path without file suffix. ", required=True)
    parser.add_argument("--o_taxo_class", type=str, help="The path of results annotation file (.txt). ", required=True)

    #3.解析参数
    args = parser.parse_args()
    seq2tensor_k3(args.i_fasta, args.i_seqchunk, args.o_seq2num, args.i_dnabert_m)
    DSNetax_classifation(args.o_seq2num, args.i_result_nums, args.o_taxo_class, args.i_DSNetax_m)

if __name__ == "__main__":
    main()