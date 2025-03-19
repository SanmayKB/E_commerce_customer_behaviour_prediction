import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

file_path = "data/ecommerce_customer_data_custom_ratios.csv"
df = pd.read_csv(file_path)

features = ["Product Price", "Quantity", "Payment Method", "Returns"]
target = "Product Category"

df = df.dropna(subset=[target] + features)

le_payment = LabelEncoder()
df["Payment Method"] = le_payment.fit_transform(df["Payment Method"])


target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])


scaler = StandardScaler()
df[["Product Price", "Quantity"]] = scaler.fit_transform(
    df[["Product Price", "Quantity"]]
)


x = df[features].values
y = df[target].values
#maxY = max(y)
#print(maxY)

#print(x[2])
#print(y[2])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

xTrainTensor = torch.tensor(x_train, dtype = torch.float32)
xTestTesnor = torch.tensor(x_test, dtype = torch.float32)
yTrainTensor = torch.tensor(y_train, dtype = torch.long)
yTestTensor = torch.tensor(y_test, dtype = torch.long)

class theoNN(nn.Module):
    def __init__(self):
        super(theoNN, self).__init__()
        self.fc1 = nn.Linear(4,8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16,4 )
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim = 1) apparently we dont need this since CrossEntropyLoss applies it
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        #self.softmax(x)
        return x

#outputSize = numClasses
model = theoNN()

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    
    outputs = model(xTrainTensor)
    loss = criterion(outputs,yTrainTensor)
    
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    if(epoch+1)%10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        
model.eval()
with torch.no_grad():
    outputs = model(xTestTesnor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == yTestTensor).sum().item()/yTestTensor.size(0)
    print(f'Accuracy:{accuracy*100:.2f}%')
        