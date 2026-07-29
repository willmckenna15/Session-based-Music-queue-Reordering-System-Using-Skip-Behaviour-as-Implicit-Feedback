import torch
import torch.nn as nn
import argparse


class DrRLLoss(nn.Module):
    def __init__(self, gamma=2.0, eta=0.1, eps=1e-1):
        super(DrRLLoss, self).__init__()
        self.gamma = gamma
        self.eta = eta  
        self.eps = eps
        self.gamma_star = gamma / (gamma - 1)
        self.c = (gamma - 1) / (eta ** (1.0 / (gamma - 1)))
        self.beta = nn.Parameter(torch.tensor(0.333))
        self.beta_optimizer = torch.optim.SGD([self.beta], lr=1e-6)

    def update_beta(self, preds, skip):
        neg_scores = preds[skip == 0].detach()
        if neg_scores.numel() == 0:
            return
        inner = self.c * torch.clamp(neg_scores - self.beta, min=0) + self.eps
        loss_beta = self.beta + torch.pow(
            torch.pow(inner, self.gamma_star).mean(),
            1.0 / self.gamma_star
        )
        self.beta_optimizer.zero_grad()
        loss_beta.backward()
        self.beta_optimizer.step()

    def forward(self, preds, skip):
        pos_scores = preds[skip == 1]
        neg_scores = preds[skip == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=preds.device)
        inner = self.c * torch.clamp(neg_scores - self.beta.detach(), min=0) + self.eps
        renyi_penalty = torch.pow(
            torch.pow(inner, self.gamma_star).mean(),
            1.0 / self.gamma_star
        )
        return -pos_scores.mean() + renyi_penalty

class PWTSLoss(nn.Module):
    def __init__(self):
        super(PWTSLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, preds, labels, song_pos, lengths, ms_played, track_length, historical_skip_rate):
        percentage_listen = ms_played / torch.clamp(track_length, min=1e-8)
        percentage_pos = song_pos / torch.clamp(lengths, min=1e-8)

        position_weight = 1 + torch.exp(-percentage_pos * 3)
        time_weight = 1 + torch.exp(-percentage_listen * 30)
        '''
        linear_part = 2 - (2 - 1.0) * (percentage_listen / 0.15)
        time_weight = torch.where(
            percentage_listen <= 0.15,
            torch.clamp(linear_part, min=1.0),
            torch.tensor(1.0, device=percentage_listen.device))
        '''
        historical_weight = 1 + torch.sigmoid((historical_skip_rate - 0.5) * 6)

        combined_weight = position_weight * time_weight * historical_weight
        combined_weight = combined_weight/combined_weight.mean()
        per_sample_loss = self.bce(preds, labels)
        return (combined_weight * per_sample_loss).mean()

def get_loss_criterion(loss_name, gamma=2.0, eta=0.1, active_scalar = 2.0):
    if loss_name == 'bce':
        return nn.BCELoss()
    elif loss_name == 'drrl':
        return DrRLLoss(gamma=gamma, eta=eta)
    elif loss_name == 'pwts':
        return PWTSLoss()
    else: 
        raise ValueError(f"Unknown loss function: {loss_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, required=True, choices=['bce', 'drrl', 'pwts'])
    parser.add_argument('--experiment', type=str, default='unnamed', help='Name for this ablation run')
    args, _ = parser.parse_known_args()
    return args