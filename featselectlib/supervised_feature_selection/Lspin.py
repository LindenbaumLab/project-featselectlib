import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


class DataSetMeta(Dataset):
    def __init__(self, data, labels=None, meta=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.meta = torch.tensor(meta, dtype=torch.float32) if meta is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None and self.meta is not None:
            return self.data[idx], self.labels[idx], self.meta[idx]
        elif self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]
        
def squared_distance(X):
    r = torch.sum(X*X, 1)
    r = r.view(-1, 1)
    D = r - 2*torch.mm(X, X.t()) + r.t()
    return D

class Lspin(nn.Module):
    def __init__(self, 
                 input_node,
                 hidden_layers_node,
                 output_node,
                 gating_net_hidden_layers_node,
                 display_step, 
                 activation_gating,
                 activation_pred,
                 feature_selection=True,
                 batch_normalization=True,
                 a=1,
                 sigma=0.5,
                 lam=0.5, 
                 gamma1=0, 
                 gamma2=0, 
                 num_meta_label=None,
                 stddev_input=0.1,
                 stddev_input_gates=0.1,
                 seed=1,
                 val=False):
        super(Lspin, self).__init__()
        
        self.a = a
        self.sigma = sigma
        self.lam = lam
        self.activation_gating = self._get_activation(activation_gating)
        self.activation_pred = self._get_activation(activation_pred)
        self.display_step = display_step
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.feature_selection = feature_selection
        self.batch_normalization = batch_normalization
        self.output_node = output_node
        self.val = val
        
        torch.manual_seed(seed)
        
        self.gating_net = self._build_gating_net(input_node, gating_net_hidden_layers_node, stddev_input_gates)
        self.prediction_net = self._build_prediction_net(input_node, hidden_layers_node, output_node, stddev_input)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'l_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'none':
            return nn.Identity()
        else:
            raise NotImplementedError('Activation function not recognized')
        
    def _build_gating_net(self, input_node, hidden_layers, stddev):
        layers = []
        prev_node = input_node
        for nodes in hidden_layers:
            layer = nn.Linear(prev_node, nodes)
            nn.init.trunc_normal_(layer.weight, std=stddev)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(self.activation_gating)
            prev_node = nodes
        layer = nn.Linear(prev_node, input_node)
        nn.init.trunc_normal_(layer.weight, std=stddev)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        layers.append(self.activation_gating) 
        return nn.Sequential(*layers)
        
    def _build_prediction_net(self, input_node, hidden_layers, output_node, stddev):
        layers = []
        prev_node = input_node
        for nodes in hidden_layers:
            layer = nn.Linear(prev_node, nodes)
            nn.init.trunc_normal_(layer.weight, std=stddev)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if self.batch_normalization:
                layers.append(nn.BatchNorm1d(nodes))
            layers.append(self.activation_pred)
            prev_node = nodes
            
        # Output layer
        layer = nn.Linear(prev_node, output_node)
        nn.init.trunc_normal_(layer.weight, std=stddev)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if self.batch_normalization:
            layers.append(nn.BatchNorm1d(output_node))
        if self.output_node != 1:  # Only add activation if it's not a regression task
            layers.append(self.activation_pred)
        return nn.Sequential(*layers)

    
    def forward(self, x, train_gates=False, compute_sim=False, Z=None):
        if self.feature_selection:
            alpha = self.gating_net(x)
            stochastic_gate = self.get_stochastic_gate_train(x, alpha, train_gates)
            x = x * stochastic_gate
            
        out = self.prediction_net(x)
        
        if self.output_node != 1:
            out = F.softmax(out, dim=1)
        
        if compute_sim and Z is not None:
            reg_sim = self.compute_similarity_penalty(stochastic_gate, Z)
        else:
            reg_sim = 0
        
        return out, alpha if self.feature_selection else None, reg_sim
    
    def get_stochastic_gate_train(self, x, alpha, train_gates):
        if train_gates:
            noise = torch.randn_like(x, device=self.device) * self.sigma
            z = alpha + noise
        else:
            z = alpha
        return self.hard_sigmoid(z, self.a)
    
    def hard_sigmoid(self, x, a):
        return torch.clamp(a * x + 0.5, 0, 1)
    
    def compute_similarity_penalty(self, stochastic_gate, Z):
        K_batch = 1.0 - squared_distance(Z) / 2.0
        D_batch = squared_distance(stochastic_gate)
        reg_sim = self.gamma1 * torch.mean(torch.mean(K_batch * D_batch, dim=-1)) + \
                  self.gamma2 * torch.mean(torch.mean((1.0 - K_batch) * -D_batch, dim=-1))
        return reg_sim

    def get_prob_alpha(self, X):
        with torch.no_grad():
            alpha = self.gating_net(X)
        return self.hard_sigmoid(alpha,self.a)

    def calculate_accuracy(self, preds, labels):
        return (preds.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
    
    def train_model(self, 
                    dataset, 
                    valid_dataset=None,
                    batch_size=64,
                    num_epoch=100,
                    lr=0.1,
                    compute_sim=False):
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion =  nn.MSELoss(reduction='mean') if self.output_node == 1 else nn.CrossEntropyLoss()
       
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if self.val and valid_dataset is not None:
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        train_losses = []
        val_losses = []
        final_val_acc = None
        
        for epoch in range(num_epoch):
            self.train()
            running_loss = 0.0
            for batch_data in train_loader:
                batch_xs, batch_ys, batch_zs = batch_data
                batch_xs, batch_ys, batch_zs = batch_xs.to(self.device), batch_ys.to(self.device), batch_zs.to(self.device)
                
                optimizer.zero_grad()
                
                pred, alpha, reg_sim = self(batch_xs, train_gates=True, compute_sim=compute_sim, Z=batch_zs)
                
                loss = criterion(pred.squeeze(), batch_ys.squeeze())
                
                if self.feature_selection:
                    reg = 0.5 - 0.5 * torch.erf((-1/(2*self.a) - alpha) / (self.sigma * np.sqrt(2)))
                    reg_gates = self.lam * torch.mean(torch.mean(reg, dim=-1))
                    loss += reg_gates + reg_sim
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            
            if self.val and valid_dataset is not None:
                # Validation
                self.eval()
                val_loss = 0.0
                val_corrects = 0
                with torch.no_grad():
                    for val_batch_data in valid_loader:
                        val_xs, val_ys, val_zs = val_batch_data
                        val_xs, val_ys, val_zs = val_xs.to(self.device), val_ys.to(self.device), val_zs.to(self.device)
                        
                        val_pred, alpha, _ = self(val_xs, train_gates=False, compute_sim=compute_sim, Z=val_zs)
                        
                        loss = criterion(val_pred.squeeze(), val_ys.squeeze())
                        
                        val_loss += loss.item() * val_xs.size(0)
                
                val_loss /= len(valid_loader.dataset)
                val_losses.append(val_loss)
                val_acc = 1 - val_loss
                final_val_acc = val_acc
                
                if (epoch + 1) % self.display_step == 0:
                    print(f"Epoch [{epoch+1}/{num_epoch}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % self.display_step == 0:
                    print(f"Epoch [{epoch+1}/{num_epoch}], Train Loss: {train_loss:.4f}")
        
        print("Training complete!")
        if self.val and valid_dataset is not None:
            print(f"Final Training Loss: {train_losses[-1]:.4f}, Final Validation Loss: {val_losses[-1]:.4f}")
        else:
            print(f"Final Training Loss: {train_losses[-1]:.4f}")
        return train_losses, val_losses, final_val_acc

    def test(self, X_test):
        """
        Predict on the test set
        """
        self.eval()
        X_test = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            pred, alpha, _ = self(X_test, train_gates=False)
            if self.output_node != 1:
                pred = pred.argmax(dim=1)
        return pred.cpu().numpy(), alpha.cpu().numpy() 

    def evaluate(self, X, y, Z, compute_sim):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)
            Z = torch.FloatTensor(Z).to(self.device)
            
            pred, alpha, reg_sim = self(X, train_gates=False, compute_sim=compute_sim, Z=Z)
            criterion =  nn.MSELoss(reduction='mean') if self.output_node == 1 else nn.CrossEntropyLoss()
            if self.output_node != 1:
                loss = criterion(pred, y.argmax(dim=1))
                acc = (pred.argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
            else:
                loss = criterion(pred.squeeze(), y.squeeze())
                acc = 1.0  # For regression, set accuracy to 1
            
        return acc, loss.item()

    def get_KD(self, X, Z):     
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X)
            Z = torch.FloatTensor(Z)
            
            K_batch_sim = 1.0 - squared_distance(Z) / 2.0       
            
            prob_alpha = self.get_prob_alpha(X)
            D_batch_sim = squared_distance(prob_alpha)
            
            K_batch_dis = squared_distance(Z) / 2.0
            
            D_batch_dis = -squared_distance(prob_alpha)
        
        return K_batch_sim.numpy(), D_batch_sim.numpy(), K_batch_dis.numpy(), D_batch_dis.numpy()
