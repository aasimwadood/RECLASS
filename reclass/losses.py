import torch
import torch.nn.functional as F

def l1_reconstruction(x, x_hat):
    return F.l1_loss(x_hat, x, reduction='mean')

def temporal_coherence_loss(preds):
    diffs = 0.0
    if preds.dim() == 5:
        g = preds.view(preds.size(0), preds.size(1), -1).mean(-1)
        diffs = F.mse_loss(g[:,1:,:], g[:,:-1,:])
    else:
        diffs = torch.tensor(0.0, device=preds.device)
    return diffs

bce = torch.nn.BCELoss(reduction='mean')
def discriminator_loss(real_scores, fake_scores):
    rlabel = torch.ones_like(real_scores)
    flabel = torch.zeros_like(fake_scores)
    return bce(real_scores, rlabel) + bce(fake_scores, flabel)

def generator_adv_loss(fake_scores):
    rlabel = torch.ones_like(fake_scores)
    return bce(fake_scores, rlabel)

def cross_modal_consistency(p_video, q_graph):
    return F.mse_loss(p_video, q_graph, reduction='mean')

def graph_structure_loss(A, A_hat):
    return F.mse_loss(A_hat, A, reduction='mean')

def explainability_loss(pred_map, pseudo_map):
    return F.l1_loss(pred_map, pseudo_map)
