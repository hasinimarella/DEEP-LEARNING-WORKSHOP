# DEEP-LEARNING-WORKSHOP

## NAME:  MARELLA HASINI
## REGISTERATION NO:212223240083
## PROGRAM:
```
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline
df = pd.read_csv('income.csv')
print(len(df))
df.head()
df.tail()
df['label'].value_counts()
cat_cols = ['sex','education','marital-status','workclass','occupation']
cont_cols = ['age','hours-per-week']
y_col = ['label']
print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')
for col in cat_cols:
    df[col] = df[col].astype('category')
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats = torch.tensor(cats, dtype=torch.long)  
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float32)

y = torch.tensor(df[y_col].values).flatten().long()  


cat_train = cats[:b-t]
cat_test  = cats[b-t:b]
con_train = conts[:b-t]
con_test  = conts[b-t:b]
y_train   = y[:b-t]
y_test    = y[b-t:b]
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = [e(x_cat[:,i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x
torch.manual_seed(33)
model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import time
start_time = time.time()
epochs = 300
losses = []

for i in range(1, epochs+1):
    model.train()
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    
    if i % 25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss Curve")
plt.show()

model.eval()
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)

print(f'CE Loss: {loss.item():.8f}')

correct = (y_val.argmax(dim=1) == y_test).sum().item()
acc = correct / len(y_test)
print(f'{correct} out of {len(y_test)} = {acc*100:.2f}% correct')
```
## OUTPUT:
<img width="841" height="208" alt="image" src="https://github.com/user-attachments/assets/017d00d4-47bf-439d-8d82-255bd783123b" />
<img width="898" height="200" alt="image" src="https://github.com/user-attachments/assets/3e68e602-2699-43ed-a587-25e83dc49f81" />
<img width="217" height="80" alt="image" src="https://github.com/user-attachments/assets/d2b80dd6-4acb-420d-8c41-cde0ca0feda0" />
<img width="212" height="63" alt="image" src="https://github.com/user-attachments/assets/c9232973-267a-43d4-8842-8909d852d74a" />
<img width="342" height="27" alt="image" src="https://github.com/user-attachments/assets/c24d48de-9e93-42e9-a738-9758d2b609b3" />
<img width="243" height="268" alt="image" src="https://github.com/user-attachments/assets/7b68699b-4c15-4fb3-8b06-1b69d2f71919" />
<img width="635" height="501" alt="image" src="https://github.com/user-attachments/assets/9d65a4b7-13b3-4768-8164-829e26dd5a2b" />


## RESULT:
Thus the binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually has been successfully executed.

