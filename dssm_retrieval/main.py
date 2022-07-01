# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import BertTokenizer

from model.Model import DSSM, BATCH_SIZE, LR, EPOCH
from reader.ImgDataLoader import ImgReaderDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import os

data_root='./data/train/'
train_path=data_root+'train/cross.train.tsv'
test_path=data_root+'test/cross.test.tsv'

def train():
    # 1、创建数据集并创立数据载入器
    train_data = ImgReaderDataset(train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test
    test_data = ImgReaderDataset(test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2、有gpu用gpu，否则cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    dssm = DSSM().to(device)
    dssm._initialize_weights()

    # 3、定义优化方式和损失函数
    optimizer = torch.optim.Adam(dssm.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (text_a, text_b, label) in enumerate(train_loader):
            # 1、把索引转化为tensor变量，载入设备，注意转化成long tensor
            a = Variable(text_a.to(device).long())
            b = Variable(text_b.to(device).long())
            l = Variable(torch.LongTensor(label).to(device))

            # 2、计算余弦相似度
            pos_res = dssm(a, b)
            neg_res = 1 - pos_res

            # 3、预测结果传给loss
            out = torch.stack([neg_res, pos_res], 1).to(device)
            loss = loss_func(out, l)

            # 4、固定格式
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 20 == 0:
                total = 0
                correct = 0
                for (test_a, test_b, test_l) in test_loader:
                    tst_a = Variable(test_a.to(device).long())
                    tst_b = Variable(test_b.to(device).long())
                    tst_l = Variable(torch.LongTensor(test_l).to(device))
                    pos_res = dssm(tst_a, tst_b)
                    neg_res = 1 - pos_res
                    out = torch.max(torch.stack([neg_res, pos_res], 1).to(device), 1)[1]
                    if out.size() == tst_l.size():
                        total += tst_l.size(0)
                        correct += (out == tst_l).sum().item()
                print('[Epoch]:', epoch + 1, '训练loss:', loss.item())
                print('[Epoch]:', epoch + 1, '测试集准确率: ', (correct * 1.0 / total))
    torch.save(dssm, './dssm.pkl')

if __name__ == '__main__':
    # train
    train()
    # test
    test_data = ImgReaderDataset(test_path)
    test_loader=DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    dssm=torch.load('./dssm.pkl').to(device)

    total = 0
    correct = 0
    TP,TN,FP,FN=0,0,0,0
    FLAG=True
    for step, (test_a, test_b, test_l) in enumerate(test_loader):
        tst_a = Variable(test_a.to(device).long())
        tst_b = Variable(test_b.to(device).long())
        tst_l = Variable(torch.LongTensor(test_l).to(device))

        pos_res = dssm(tst_a, tst_b)
        neg_res =  - pos_res
        sta=torch.stack([neg_res, pos_res], 1).to(device)
        out = torch.max(sta, dim=1)[1]

        total += tst_l.size(0)
        correct += (out == tst_l).sum().item()

        #计算精确率、召回率
        TP += ((out == 1) & (tst_l == 1)).sum().item()
        TN += ((out == 0) & (tst_l == 0)).sum().item()
        FN += ((out == 0) & (tst_l == 1)).sum().item()
        FP += ((out == 1) & (tst_l == 0)).sum().item()

        if FLAG == True:
            for i in range(30,40):
                a, b, l = test_data[i][0], test_data[i][1], test_data[i][2]
                print('标签：',l,'预测：',out[i].item())
        FLAG=False

    p = TP / (TP + FP)
    r = TP / (TP + FN)

    print('测试集准确率: ', (correct * 1.0 / total))
    print('测试集精确率：', p)
    print('测试集召回率：', r)
    print('测试集f1-score：', 2 * r * p / (r + p))