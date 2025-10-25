import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=32, temporal_pool='avg'):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters*2)
        self.conv3 = ConvBlock(base_filters*2, base_filters*4)
        self.temporal_pool = temporal_pool
        self.proj = nn.Linear(base_filters*4, base_filters*4)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, 1).view(B, T, -1)
        if self.temporal_pool == 'avg':
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]
        z = self.proj(x)
        return z

class FlowEncoder(SpatioTemporalEncoder):
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__(in_channels, base_filters)

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=32, hidden=64, out_dim=128):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU(inplace=True)
    def forward(self, node_feats, adj):
        h = self.act(self.lin1(node_feats))
        h = torch.bmm(adj, h)
        h = self.lin2(h.mean(dim=1))
        return h

class SpatioTemporalGenerator(nn.Module):
    def __init__(self, latent_dim=256, base_filters=64, out_channels=3, clip_len=32, img_h=256, img_w=448):
        super().__init__()
        self.clip_len = clip_len
        self.fc = nn.Linear(latent_dim, base_filters*8*8)
        self.deconv1 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(base_filters*2, base_filters, 4, 2, 1)
        self.final_conv = nn.Conv2d(base_filters, out_channels, 3, 1, 1)
    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, -1, 1, 1)
        x = F.interpolate(x, size=(8, 14), mode='nearest')
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        img = torch.sigmoid(self.final_conv(x))
        img = img.unsqueeze(2).repeat(1,1,self.clip_len,1,1)
        img = img.permute(0,2,1,3,4)
        return img

class GraphGenerator(nn.Module):
    def __init__(self, latent_dim=128, node_feat_dim=64, max_nodes=40):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_nodes*node_feat_dim)
        )
        self.edge_att = nn.Linear(2*node_feat_dim, 1)
    def forward(self, z):
        B = z.size(0)
        node_feats = self.node_mlp(z).view(B, self.max_nodes, -1)
        a_i = node_feats.unsqueeze(2).repeat(1,1,self.max_nodes,1)
        a_j = node_feats.unsqueeze(1).repeat(1,self.max_nodes,1,1)
        pair = torch.cat([a_i, a_j], dim=-1)
        e_logits = self.edge_att(pair).squeeze(-1)
        A_hat = torch.sigmoid(e_logits)
        return node_feats, A_hat

class SpatioTemporalDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(32, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(64, 128, 3, 2, 1)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        h = self.conv(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(h))

class GraphDiscriminator(nn.Module):
    def __init__(self, node_feat_dim=64):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(node_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, 1)
    def forward(self, node_feats, adj):
        h = self.lin(node_feats).mean(dim=1)
        return torch.sigmoid(self.classifier(h))
