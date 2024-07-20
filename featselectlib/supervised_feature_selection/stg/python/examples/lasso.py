import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, learning_rate=0.01, batch_size=64, lam=0.0, device='cpu'):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=lam)
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.lam = lam
        self.device = device
    
    def forward(self, x):
        return self.linear(x)
    
    def get_dataloader(self, X, y, shuffle=True):
        X_tensor = torch.tensor(X).float()  
        y_tensor = torch.tensor(y).float()
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

        
    def train_model(self, train_loader, val_loader=None):
        self.train()
        train_loss = 0
        train_acc=0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(X_batch)
            outputs = outputs.squeeze(1)
            l1_reg = sum(torch.abs(param).sum() for param in self.parameters())
            loss = self.criterion(outputs, y_batch)  
            total_loss = loss + self.lam * l1_reg

            preds = torch.sigmoid(outputs) >= 0.5
            train_acc += (preds == y_batch).float().mean().item()

            total_loss.backward()
            self.optimizer.step()
            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_loss = None
        if val_loader is not None:
            val_loss,val_acc = self.validate(val_loader)  
            
        return train_loss, val_loss,train_acc,val_acc

    def validate(self, val_loader):
        self.eval()  
        val_loss = 0
        val_acc=0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.forward(X_batch)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, y_batch)
                preds = torch.sigmoid(outputs) >= 0.5
                val_acc += (preds == y_batch).float().mean().item()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        return val_loss,val_acc

    def fit(self, X_train, y_train, nr_epochs, valid_X=None, valid_y=None, verbose=True, print_interval=100, shuffle=True):
        train_loader = self.get_dataloader(X_train, y_train, shuffle)
        val_loader = self.get_dataloader(valid_X, valid_y, shuffle=False) 
        train_losses=[]
        val_losses=[]
        train_accuracy=[]
        val_accuracy=[]
        weights=[]
       
        for epoch in range(nr_epochs):
            train_loss, val_loss,train_acc,val_acc = self.train_model(train_loader, val_loader)  # Training and validation in one call
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)
            # Tracking the wieghts
            weights.append(self.linear.weight.data.clone())

            if verbose and (epoch + 1) % print_interval == 0:
                message = f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}'
                if val_loss is not None : print(message + f', Val Loss: {val_loss:.4f}')
                else: print(message)
        return train_losses,val_losses,train_accuracy,val_accuracy,weights


    def predict(self, X):
        self.eval()  
        with torch.no_grad(): 
            X_tensor = torch.tensor(X).float().to(self.device)
            outputs = self.forward(X_tensor)
            predictions = torch.sigmoid(outputs)  
            predicted_classes = predictions.round()  
        return predicted_classes.cpu().numpy()
