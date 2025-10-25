import torch
from torch import optim
from torch.utils.data import DataLoader
from reclass.models import (SpatioTemporalEncoder, FlowEncoder, GraphEncoder,
                            SpatioTemporalGenerator, GraphGenerator,
                            SpatioTemporalDiscriminator, GraphDiscriminator)
from reclass.losses import *
from reclass.utils import ema_update
import torch.nn.functional as F

class Trainer:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        latent_st = 256
        latent_g = 128
        self.est = SpatioTemporalEncoder(in_channels=3).to(self.device)
        self.eflow = FlowEncoder(in_channels=2).to(self.device)
        self.eg = GraphEncoder(in_dim=32, out_dim=latent_g).to(self.device)
        self.gst = SpatioTemporalGenerator(latent_dim=latent_st, clip_len=cfg['train']['clip_len']).to(self.device)
        self.gg = GraphGenerator(latent_dim=latent_g).to(self.device)
        self.dst = SpatioTemporalDiscriminator(in_channels=3).to(self.device)
        self.dg = GraphDiscriminator().to(self.device)
        params = list(self.est.parameters()) + list(self.eflow.parameters()) + list(self.gst.parameters()) + list(self.eg.parameters()) + list(self.gg.parameters())
        self.optG = optim.Adam(params, lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']))
        self.optD = optim.Adam(list(self.dst.parameters()) + list(self.dg.parameters()), lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']))
        self.ema_score = None
        self.ema_beta = cfg['train'].get('ema_beta', 0.9)
        self.dataloader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    def train_epoch(self, epoch):
        self.est.train(); self.eflow.train(); self.eg.train()
        self.gst.train(); self.gg.train(); self.dst.train(); self.dg.train()
        for batch in self.dataloader:
            rgb = batch['rgb'].to(self.device)
            flow = batch['flow'].to(self.device)
            nodes = batch['nodes'].to(self.device)
            adj = batch['adj'].to(self.device)
            zst_rgb = self.est(rgb)
            zst_flow = self.eflow(flow)
            zst = torch.cat([zst_rgb, zst_flow], dim=-1)
            zg = self.eg(nodes, adj)
            x_hat = self.gst(zst)
            node_hat, A_hat = self.gg(zg)
            real_score_st = self.dst(rgb)
            fake_score_st = self.dst(x_hat.detach())
            real_score_g = self.dg(nodes, adj)
            fake_score_g = self.dg(node_hat.detach(), A_hat.detach())
            ld_st = discriminator_loss(real_score_st, fake_score_st)
            ld_g = discriminator_loss(real_score_g, fake_score_g)
            ld_total = ld_st + ld_g
            self.optD.zero_grad()
            ld_total.backward()
            self.optD.step()
            lrec = l1_reconstruction(rgb, x_hat)
            ladd = generator_adv_loss(self.dst(x_hat))
            lflow_reg = temporal_coherence_loss(x_hat)
            lgraph_struct = graph_structure_loss(adj, A_hat)
            lcons = cross_modal_consistency(zst, zg)
            lexp = torch.tensor(0., device=rgb.device)
            Lgst = 1.0 * ladd + 1.0 * lrec + 0.1 * lflow_reg + 0.5 * lcons + 0.2 * lexp
            Lgg = 1.0 * generator_adv_loss(self.dg(node_hat, A_hat)) + 1.0 * lgraph_struct + 0.5 * lcons
            self.optG.zero_grad()
            (Lgst + Lgg).backward()
            self.optG.step()
            gamma1, gamma2, gamma3 = 1.0, 1.0, 1.0
            rec_term = F.mse_loss(rgb, x_hat, reduction='none').view(rgb.size(0), -1).mean(dim=1)
            dst_term = 1.0 - self.dst(x_hat).view(-1)
            dg_term = 1.0 - self.dg(node_hat, A_hat).view(-1)
            A_t = gamma1*rec_term + gamma2*dst_term + gamma3*dg_term
            if self.ema_score is None:
                self.ema_score = A_t.detach()
            else:
                self.ema_score = ema_update(self.ema_score, A_t.detach(), self.ema_beta)
        print(f"Epoch {epoch} done; last EMA score mean: {self.ema_score.mean().item():.4f}")
    def train(self):
        for e in range(self.cfg['train']['epochs']):
            self.train_epoch(e)
