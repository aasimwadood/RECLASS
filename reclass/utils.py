import torch
import numpy as np

def ema_update(old, new, beta):
    return beta*old + (1.-beta)*new

def grad_cam_on_discriminator(discriminator, x, score):
    # Very simplified illustrative Grad-CAM. For research use, register hooks on chosen conv layers.
    grads = torch.autograd.grad(score, x, retain_graph=True, create_graph=False)[0]
    weights = grads.mean(dim=[2,3,4], keepdim=True) if grads.dim()==5 else grads.mean(dim=[2,3], keepdim=True)
    cam = (weights * x).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam / (cam.max()+1e-8)
    return cam.detach()
