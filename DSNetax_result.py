import argparse
import sys

import joblib
import torch
from DSNetax_model import ResnNeSt, Bottleneck
import torch.utils.data as Data
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

enc_path = 'label_model.pkl'
enc = joblib.load(enc_path)

def DSNetax_classifation(seq2num, result_nums, taxo_class, DSNetax_path):
    model = ResnNeSt(Bottleneck, [2, 3, 2, 2],
                     radix=2, groups=1, bottleneck_width=32,
                     deep_stem=True, stem_width=32, avg_down=True,
                     avd=True, avd_first=False)

    model_path = DSNetax_path
    print(torch.cuda.is_available())

    torch.cuda.current_device()
    torch.cuda._initialized = True
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    for i in range(100):#

        if os.path.exists(seq2num + str(i) + '.pth'):

            rpath = taxo_class
            f1 = open(rpath, 'a+', newline='', encoding='utf-8')
            csv_writer = csv.writer(f1)

            x_test = torch.load(seq2num + str(i) + '.pth').float()

            valid_dataset = Data.TensorDataset(x_test)
            valid_loader = Data.DataLoader(
                dataset=valid_dataset,  # 数据，封装进Data.TensorDataset()类的数据
                batch_size=8,  # 每块的大小
                shuffle=False,  # 要不要打乱数据 (打乱比较好)
                num_workers=4,  # 多进程（multiprocess）来读数据
            )
            model.eval()

            if result_nums == 1:
                for p, (input) in enumerate(valid_loader):

                    input = input[0].cuda()
                    output = model(input)
                    # 获取最高概率对应的类别

                    predicted_labels = torch.argmax(output, dim=-1)
                    prr = enc.inverse_transform(predicted_labels.cpu().numpy())
                    for line in prr:

                        csv_writer.writerow([line.strip('\n')])
            elif result_nums == 5:

                for q, (input, target) in enumerate(valid_loader):

                    input = input.cuda()
                    output = model(input)
                    # 获取最高概率对应的类别

                    predicted_labels = torch.topk(output, k=5, largest=True, dim=-1, sorted=True)

                    for j in predicted_labels:
                        result_list = []
                        for m in j:
                            prr = enc.inverse_transform(m.cpu().numpy())
                            result_list.append(prr)
                            csv_writer.writerow([result_list.strip('\n')])
            f1.close()
        else:
            break
def main():
    #1 使用的第一步argparse是创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='manual to this script')
    #2add_argument()添加参数。
    parser.add_argument("--i_seq2num", type=str, help="Processed file path without file suffix. ", required=True)
    parser.add_argument("--i_result_nums", type=str, help="Number of output results. ", default=1, choices=['1', '5'], required=False)
    parser.add_argument("--o_taxo_class", type=str, help="The path of results annotation file (.txt). ", required=True)
    parser.add_argument("--i_DSNetax_m", type=str, help="The DSNetax model file path. ", default='trained_DSNetax.pth', required=False)
    #3.解析参数
    args = parser.parse_args()
    DSNetax_classifation(args.i_seq2num, args.i_result_nums, args.o_taxo_class, args.i_DSNetax_m)

if __name__ == "__main__":
    main()
