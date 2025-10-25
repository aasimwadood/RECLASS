import torch
from reclass.utils import grad_cam_on_discriminator

def infer_step(model_dict, batch, device, fusion_alpha=0.85, ema_prev=None, beta=0.9):
    est, eflow, eg, gst, gg, dst, dg = [model_dict[k] for k in ['est','eflow','eg','gst','gg','dst','dg']]
    rgb = batch['rgb'].to(device)
    flow = batch['flow'].to(device)
    nodes = batch['nodes'].to(device)
    adj = batch['adj'].to(device)
    zst = torch.cat([est(rgb), eflow(flow)], dim=-1)
    zg = eg(nodes, adj)
    x_hat = gst(zst)
    node_hat, A_hat = gg(zg)
    rec_term = ((rgb - x_hat)**2).view(rgb.size(0), -1).mean(dim=1)
    sst = dst(x_hat).view(-1)
    sgt = dg(node_hat, A_hat).view(-1)
    gamma1,gamma2,gamma3 = 1.0,1.0,1.0
    At = gamma1*rec_term + gamma2*(1.0 - sst) + gamma3*(1.0 - sgt)
    A_fusion = At
    if ema_prev is None:
        ema = A_fusion.detach()
    else:
        ema = beta*ema_prev + (1-beta)*A_fusion.detach()
    dst_score = dst(x_hat)
    cam = grad_cam_on_discriminator(dst, x_hat, dst_score.sum())
    return {'A': A_fusion.detach(), 'A_ema': ema, 'cam': cam}
