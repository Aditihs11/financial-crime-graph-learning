import torch
import torch.nn as nn
from copy import deepcopy

class EWC(object):
    """
    Elastic Weight Consolidation (EWC) for Continual Graph Learning.
    Calculates the Fisher Information Matrix to protect important weights of past tasks
    when training on new data (e.g., new fraud typologies).
    """
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        
        # Save current parameters as means for the EWC penalty
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data
            
        self._precision_matrices = self._diag_fisher()
        
    def _diag_fisher(self):
        """
        Computes the diagonal Fisher Information Matrix.
        Since this is a node classification task, we compute Fisher on the account nodes.
        """
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        
        # Dummy data extraction based on our heterogeneous graph
        x_dict = {k: v.to(self.device) for k, v in self.dataset.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in self.dataset.edge_index_dict.items()}
        y = self.dataset['account'].y.to(self.device)
        mask = self.dataset['account'].train_mask.to(self.device)

        self.model.zero_grad()
        out = self.model(x_dict, edge_index_dict)
        
        # Calculate log likelihood using CrossEntropy
        loss = nn.CrossEntropyLoss()(out[mask], y[mask])
        loss.backward()

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # E[ (\nabla \log p)^2 ]
                precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        """
        Computes the EWC penalty for the current model weights against the previous task.
        """
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
