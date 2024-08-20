import torch.utils.data as Data
import sys
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import pandas as pd
import gc
import shutil
from DSNetax_model import ResnNeSt, Bottleneck
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 1, 2"


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def accuracy(output, target, topk=(1,)):
    """
        计算topk的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        根据 is_best 存模型，一般保存 valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        训练代码
        参数：
            train_loader - 训练集的 DataLoader
            model - 模型
            criterion - 损失函数
            optimizer - 优化器
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #判断最后一块是不是只有一条数据
        if len(target) == 1:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        #print(loss)

        # measure accuracy and record loss
        [prec1, prec5, prec10], class_to = accuracy(output, target, topk=(1, 5, 10))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        top10.update(prec10[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Prec@10 {top10.val:.3f} ({top10.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, top10=top10))
        # writer.add_scalar('loss/train_loss', losses.val, global_step=i)
    writer.add_scalar('loss/train_loss', losses.avg, global_step=(epoch + 1))
    writer.add_scalar('acc/train_acc', top1.avg, global_step=(epoch + 1))


def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if len(target) == 1:
                break
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, prec5, prec10], class_to = accuracy(output, target, topk=(1, 5, 10))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            top10.update(prec10[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Prec@10 {top10.val:.3f} ({top10.avg:.3f})'.format(
                    phase, i, len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1, top5=top5, top10=top10))
            # writer.add_scalar('loss/valid_loss', losses.val, global_step=i)
        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Prec@10 {top10.avg:.3f}'
              .format(phase, top1=top1, top5=top5, top10=top10))
    writer.add_scalar('loss/valid_loss', losses.avg, global_step=(epoch + 1))
    writer.add_scalar('acc/valid_acc', top1.avg, global_step=(epoch + 1))

    return top1.avg, top5.avg, top10.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # 将print保存在txt中
    pd.set_option('display.max_rows', None)
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('loss-0613k3_DP_resnest.txt')

    y_train = torch.load('/home/zhaohongyuan/DL-DNA/silva_data/train_data/bac_sv_tax_label_train_tsor.pth')

    y_test = torch.load('/home/users-data/liukai/zhaohongyuan_k3tes/test_data/bac_sv_tax_label_test_tsor.pth')

    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    model = ResnNeSt(Bottleneck, [2, 3, 2, 2],
                   radix=2, groups=1, bottleneck_width=32,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    #print(model)
    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    lr_init = 0.001
    lr_stepsize = 10
    weight_decay = 0.001
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)

    writer = SummaryWriter(comment='-resnest')
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    epochs = 60
    best_prec1 = 0
    for epoch in tqdm(range(epochs)):
        gc.collect()
        torch.cuda.empty_cache()
        for i in range(0, len(y_train), 10000):
            path_str = '/home/zhaohongyuan/DL-DNA/silva_data/train_data/k3/bac_sv_seqs_train_k3_' + str(i) + '.pth'
            x_train = torch.load(path_str).float()
            #print(x_train.shape)
            y_train1 = y_train[i: i + 10000]
            y_train1 = y_train1.long()
            y_train1 = torch.flatten(y_train1, 0)
            #print(y_train1.shape)

            train_dataset = Data.TensorDataset(x_train, y_train1)
            train_loader = Data.DataLoader(
                dataset=train_dataset,  # 数据，封装进Data.TensorDataset()类的数据
                batch_size=32,  # 每块的大小
                shuffle=True,  # 要不要打乱数据 (打乱比较好)
                num_workers=2,  # 多进程（multiprocess）来读数据
            )

            # scheduler.step()
            train(train_loader, model, criterion, optimizer, epoch, writer)

        # 在验证集上测试效果:这里的验证集由于内存问题拆分为多个文件
        for i in range(0, len(y_test), 10000):
            path_str_x = '/home/users-data/liukai/zhaohongyuan_k3tes/k3/bac_sv_seqs_test_k3_' + str(i) + '.pth'
            x_test = torch.load(path_str_x).float()
            y_test1 = y_test[i: i + 10000]
            y_test1 = y_test1.long()
            y_test1 = torch.flatten(y_test1, 0)

            valid_dataset = Data.TensorDataset(x_test, y_test1)
            valid_loader = Data.DataLoader(
                dataset=valid_dataset,  # 数据，封装进Data.TensorDataset()类的数据
                batch_size=32,  # 每块的大小
                shuffle=False,  # 要不要打乱数据 (打乱比较好)
                num_workers=4,  # 多进程（multiprocess）来读数据
            )

            valid_prec1, valid_prec5, valid_prec10 = validate(valid_loader, model, criterion, epoch, writer, phase="VAL")
            # writer.add_scalars(main_tag='Metrics', tag_scalar_dict={'ValLoss': val_loss,'RMSE': rmse}, global_step=epoch)

        is_best = valid_prec1 > best_prec1
        best_prec1 = max(valid_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnest',
            'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='checkpoint_resnest_0524k3_DP.pth')
    writer.close()
