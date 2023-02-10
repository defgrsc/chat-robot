import torch        #神經網路模型函式庫
import torch.nn as nn

class NeuralNet(nn.Module):    
    def __init__(self, input_size, hidden_size, num_classes): 
        super(NeuralNet, self).__init__()   #繼承nn.Module
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):   #x為資料
        # 輸入層
        out = self.l1(x)    
        out = self.relu(out)    #進到下一層之前都要進到激勵函數    
       
        # 隱藏層
        out = self.l2(out)
        out = self.relu(out)
       
        # 輸出層
        out = self.l3(out)
       
        # 最後不用激勵函數 (引用ReLU()使分類器效果更好)
        return out