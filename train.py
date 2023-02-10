import json     #資料呈現為JavaScript格式 函式庫
import numpy as np
import torch    #神經網路模型函式庫
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #神經網路模型
from model import NeuralNet  #輸入隱藏輸出層
from nltk_utils import tokenize, stem, bag_of_words #語言處理

with open('intents.json','r') as f:   #開啟的檔案在with的範圍內才可使用 
    intents_box = json.load(f)        # r 以只讀方式打開文件
    
all_words = []  #儲存所有文字
tags =[]    #儲存標籤 情境
xy = []     #儲存樣式與標籤  (x :文字 y:情境tag)

for intent in intents_box['intents_list']:
    tag = intent['tag']
    tags.append(tag)  #加一個元素在串列尾端

    for sentence in intent['patterns']: #patterns:接收的話
        w = tokenize(sentence)
        all_words.extend(w) #extend 一次加多個元素在串列尾端
        xy.append((w,tag))  #tuple
        
ignore_words = ['?','!','.',',','~','&',':',';',"%"]
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))  #sorted()將字母排列 
tags = sorted(set(tags))            #set()將串列中重複的字刪除

#-----Training Data-----#
X_train = [] #輸入: BOW向量
y_train = [] #輸出: 儲存tag(分類)

for (sentence, tag) in xy:
    bag = bag_of_words(sentence,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
#-----Training Data-----#

# print(type(X_train),'\n',X_train)
# print(type(y_train),'\n',y_train)

#--- Pytorch神經網路設定區域 ---#
#Hyperparameter(超參數)
batch_size = 8                 #每次拿多少筆資料訓練
input_size = len(X_train[0])    #輸入層大小
hidden_size = 7                 #隱藏層大小
output_size = len(tags)         #輸出層大小
learning_rate = 0.0005          #學習率(每次改進)
num_epochs = 8000               #訓練多少次

#創建pytorch數據集
class ChatDataset(Dataset):
    #初始化函式
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    #用序號(index)取得資料
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    #取得training set (資料集)大小
    def __len__(self):
        return self.n_samples
#--- Pytorch神經網路設定區域 ---#

#--- Pytorch神經網路訓練區域 ---#
def main():
    # 模型、數據集、硬體整合
    dataset = ChatDataset()
    train_loader  = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) #數據訓練加載器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                           #設定用甚麼硬體來訓練模型 (cuda指gpu)
    model = NeuralNet(input_size, hidden_size, output_size).to(device)                              #把神經網路模型NeuralNet() 用.to(device)複製到硬體

    #優化準則(損失函數)
    criterion = nn.CrossEntropyLoss()   
    #優化器(根據學習率來調整模型參數)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    for epoch in range(num_epochs):
        train_loss = 0.0    #損失函數數值歸零
        for (sentence, tag) in train_loader:
            #梯度歸零
            optimizer.zero_grad()   
            
            sentence = sentence.to(device)  
            tag = tag.to(dtype=torch.long).to(device)
            
            # 前向傳播(forward propagation) 
            # 將輸入進到模型運算得出損失函數數值
            outputs = model(sentence)
            loss = criterion(outputs, tag)
            
            # 反向傳播(backward propagation)
            # 利用損失函數數值和學習率調整權重
            loss.backward()
            
            # 更新所有參數
            optimizer.step()
            
        #每100次迭代，把損失函數數值顯示出來
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            
    print(f'final loss, loss={loss.item():.4f}')
    
    # 將訓練完的資料、分類器儲存起來，存在data.pth這個檔案裡
    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }
    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. File saved to {FILE}')
#--- Pytorch神經網路訓練區域 ---#

if __name__=="__main__":
    main()