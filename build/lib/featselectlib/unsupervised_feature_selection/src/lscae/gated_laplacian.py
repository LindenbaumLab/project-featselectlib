import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np


class GatedLaplacianModel(nn.Module):
    def __init__(self, input_dim=12, seed=1, lam=0.1, fac=2, knn=2, is_param_free_loss=True,
                 num_epochs=100, batch_size=64, learning_rate=1e-3, verbose=True,print_interval=100):
        super(GatedLaplacianModel, self).__init__()
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.lam = lam
        self.fac = fac
        self.knn = knn
        self.sigma = 0.5  
        self.is_param_free_loss = is_param_free_loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.print_interval=print_interval
        self.alpha = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate,momentum=0)

    def forward(self, x):
        masked_input = self.feature_selector(x)
        L = self.full_affinity_knn_pt(masked_input, knn=self.knn, fac=self.fac)
        return L, masked_input
    

    def train_step(self, x):
        self.optimizer.zero_grad()
        L, masked_input = self.forward(x)
        L2 = torch.matmul(L, L)
        laplacian_score= -torch.mean(torch.mm(L2, masked_input) * masked_input)
        reg = 0.5 - 0.5 * torch.erf((-0.5-self.alpha) / (self.sigma * torch.sqrt(torch.tensor(2.0))))
        reg_gates = reg.mean() + 1e-6

        if self.is_param_free_loss:
            loss = laplacian_score / reg_gates
        else:
            loss = laplacian_score + self.lam * reg_gates
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), laplacian_score.item(), reg_gates.item()
    
    def feature_selector(self, x):
        base_noise = torch.randn(self.alpha.size()).normal_()
        z = self.alpha.unsqueeze(0) + self.sigma * base_noise
        stochastic_gate = self.hard_sigmoid(z)  
        masked_x = x * stochastic_gate
        return masked_x

    def select_features(self, dataloader):
        for epoch in range(self.num_epochs):
            batch_losses = []
            laplacian_scores = []
            for batch_x, in dataloader:
                batch_x = batch_x.to(next(self.parameters()).device) 
                loss, laplacian_score,reg_gates = self.train_step(batch_x)
                batch_losses.append(loss)
                laplacian_scores.append(laplacian_score)
                                   
            epoch_loss = np.mean(batch_losses)
            epoch_laplacian_score = np.mean(laplacian_scores)
            if self.verbose == True:
                if epoch % self.print_interval == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, "
                    f"Laplacian Score: {epoch_laplacian_score:.4f}",f"reg: {reg_gates}")
                if epoch % (2*self.print_interval)==0:
                    print('Selection probs: \n ', self.get_gates('prob'), '\n')

        probs = torch.tensor(self.get_gates("prob"))
        max_value = probs.max()
        # Find all indices where the value equals the maximum value
        max_indices = (probs == max_value).nonzero(as_tuple=True)[0]
        selected_features = set(max_indices.tolist())
        return selected_features
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)
    
    def get_gates(self, mode):
        if mode == 'raw':
            return self.alpha.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.alpha.detach().cpu().numpy() + 0.5)) 
        
    def squared_distance_pt(self,X):
        r = torch.sum(X * X, 1)
        r = r.reshape(-1, 1)
        D = r - 2 * torch.matmul(X, X.t()) + r.t()
        return D

    
    def full_affinity_knn_pt(self, X, knn=2, fac=0.6):
        Dx = self.squared_distance_pt(X)
        # Get the distances of the k-nearest neighbors directly
        sorted_distances, indices = torch.sort(Dx, dim=1)
        # You should select the k-th nearest, not -Dx as it was wrongly mentioned
        knn_distances = sorted_distances[:, knn-1]
        mu, ml = self.calculate_percentiles(knn_distances)
        sigma = (mu + ml) / 2.
        sigma = torch.where(sigma < 1e-8, torch.tensor(1.0, device=X.device), sigma)
        W = torch.exp(-Dx / (fac * sigma))
        Dsum = torch.sum(W, dim=1)
        Dminus = torch.pow(Dsum, -1)
        Dminus = torch.diag_embed(Dminus)
        P = torch.matmul(Dminus, W)
        return P

    def calculate_percentiles(self, distances, percentile=50.0):
        # This ensures that we always pick the higher and lower medians correctly.
        sorted_distances = torch.sort(distances).values
        index = int(percentile / 100.0 * len(sorted_distances))
        if len(sorted_distances) % 2 == 0:
            # Even number of samples: average the two middle elements
            mu = sorted_distances[index]  # higher
            ml = sorted_distances[index - 1]  # lower
        else:
            # Odd number of samples: middle element is both mu and ml
            mu = ml = sorted_distances[index]
        return mu, ml
