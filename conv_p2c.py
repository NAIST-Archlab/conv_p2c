# pytorch pth file to c language syntax
import torch
import torch.nn as nn
import numpy as np
import argparse

# 再帰的配列ダンプ(次元が分からなくても動作する)
# reference: https://msyksphinz.hatenablog.com/entry/2018/03/24/040000
def recurse_dump(array, dim=0):
    if array.ndim == 1:
        for tab in range(dim):
            print(" ", end="")
        print("{", end="")
        for i in range(len(array)):
            print("%10.20f " % array[i], end="") # 桁数はここでいじる
            if i != len(array)-1:
                print(",", end="")
        for tab in range(dim):
            print(" ", end="")
        print("}", end="")
    else:
        for tab in range(dim):
            print(" ", end="")
        print("{")
        for i in range(len(array)):
            array_elem = array[i]
            recurse_dump(array_elem, dim+1)
            if i != len(array)-1:
                print(",")
            else:
                print("")
        for tab in range(dim):
            print(" ", end="")
        print(" }", end="")

def recurse_dump_file(f, array, dim=0):
    if array.ndim == 1:
        for tab in range(dim):
            print(" ", end="", file=f)
        print("{", end="", file=f)
        for i in range(len(array)):
            print("%10.20f " % array[i], end="", file=f) # 桁数はここでいじる
            if i != len(array)-1:
                print(",", end="", file=f)
        for tab in range(dim):
            print(" ", end="", file=f)
        print("}", end="", file=f)
    else:
        for tab in range(dim):
            print(" ", end="", file=f)
        print("{", file=f)
        for i in range(len(array)):
            array_elem = array[i]
            recurse_dump_file(f, array_elem, dim+1)
            if i != len(array)-1:
                print(",", file=f)
            else:
                print("")
        for tab in range(dim):
            print(" ", end="", file=f)
        print(" }", end="", file=f)

# load to cpu
device = torch.device('cpu')

# Define Hyper-parameters for model
L = 14
input_size = L * L
hidden_size = 80
num_classes = 10

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ############ pthから多次元配列を格納したtxtに変換 ############
# .cを生成する
# Conv2dレイヤーありのネットワークでも動作確認済み
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# pthファイルをロード
model.load_state_dict(torch.load('model_sample.pth'))
save_path = "./"

# .c ファイルのオープン
f = open('nn_parameter.c', 'w', encoding='utf-8')

# print("Network Architecture & Patameter Size:")
for key, param in model.state_dict().items(): #この文で、keyの中のparam(weightとbias)ごとに処理ができる。
    # もともとのpthの当該keyの中のparamのshapeを確認
    print(key, "\n", param.size())
    # 中身の値を確認したいときはこっちを有効化
    # print(key, "\n", param.size(), "\n", param)

    # paramをpytorch tensor型からnumpyのndarrayに変換(余計な勾配情報を捨てる)
    print("Typecasting")
    param_np = param.numpy()
    key_str = str(key)
    key_str = key_str.replace('.', '_') # weightとbiasをc言語のほうでネットワークレイヤーを定義する構造体のメンバにしておけばこの行はいらない。
    dim_str = ''
    for i in range(param_np.ndim):
        dim_str = dim_str + "[" + str(param_np.shape[i]) +"]"
    key_str = key_str + dim_str
    print("Type:", type(param_np),"\t", "Dimension:", param_np.ndim, "\t", "Shape:", param_np.shape)
    print("Displaying as C language style")
    print("const float", key_str, "=")
    recurse_dump(param_np)
    key_str = "const float " + key_str + " = "
    f.write(key_str)
    f.write('\n')
    recurse_dump_file(f, param_np)
    f.write(";")
    f.write('\n')

f.close()