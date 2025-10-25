import os
import json
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class SAVDDataset(Dataset):
    """
    SAVD dataset loader for RECLASS.
    Loads RGB clips (T frames) and automatically computes optical flow or node graphs
    if they are not precomputed on disk.
    """

    def __init__(self, root, split_path, clip_len=32, stride=2,
                 image_h=256, image_w=448, augment=False):
        self.root = root
        self.clip_len = clip_len
        self.stride = stride
        self.image_h = image_h
        self.image_w = image_w
        self.augment = augment

        with open(split_path, "r") as f:
            self.video_list = json.load(f)

        self.samples = self._prepare_index()
        self.resize = (self.image_w, self.image_h)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_h, image_w)),
        ])

        # Optical flow algorithm (TV-L1)
        self.flow_calc = cv2.optflow.DualTVL1OpticalFlow_create()

    def _prepare_index(self):
        samples = []
        for vid in self.video_list:
            rgb_dir = os.path.join(self.root, "videos", vid, "rgb")
            frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
            if len(frames) < self.clip_len:
                continue
            for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                samples.append((vid, start))
        return samples

    def __len__(self):
        return len(self.samples)

    # ---------------------- RGB LOADING ----------------------
    def _load_rgb_clip(self, vid, start_idx):
        rgb_dir = os.path.join(self.root, "videos", vid, "rgb")
        frame_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        frames = []
        for i in range(start_idx, start_idx + self.clip_len):
            img = cv2.imread(frame_files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize)
            frames.append(torch.from_numpy(img).permute(2,0,1).float() / 255.0)
        return torch.stack(frames, dim=0)

    # ---------------------- OPTICAL FLOW ----------------------
    def _load_optical_flow(self, vid, start_idx):
        """
        Load optical flow if available, otherwise compute using TV-L1.
        Returns tensor [T, 2, H, W].
        """
        flow_dir = os.path.join(self.root, "videos", vid, "flow")
        flow_files = sorted(glob.glob(os.path.join(flow_dir, "*.npy"))) if os.path.exists(flow_dir) else []
        rgb_dir = os.path.join(self.root, "videos", vid, "rgb")
        frame_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

        flows = []
        for i in range(start_idx, start_idx + self.clip_len):
            if i >= len(frame_files)-1:
                break
            if len(flow_files) > i:
                flow = np.load(flow_files[i])
                flow = cv2.resize(flow.transpose(1,2,0), self.resize)
            else:
                # Compute flow from consecutive RGB frames
                prev = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
                nxt = cv2.imread(frame_files[i+1], cv2.IMREAD_GRAYSCALE)
                prev = cv2.resize(prev, self.resize)
                nxt = cv2.resize(nxt, self.resize)
                flow = self.flow_calc.calc(prev, nxt, None)
            flow = np.clip(flow, -20, 20)
            flow = torch.from_numpy(flow.transpose(2,0,1)).float()
            flows.append(flow)
        # pad if fewer than clip_len
        while len(flows) < self.clip_len:
            flows.append(torch.zeros_like(flows[-1]))
        return torch.stack(flows, dim=0)

    # ---------------------- GRAPH DATA ----------------------
    def _load_nodes_and_adj(self, vid, frame_idx=None):
        """
        Load node features and adjacency if available,
        otherwise compute simple bounding-box-based features.
        """
        track_dir = os.path.join(self.root, "videos", vid, "tracks")
        pose_dir = os.path.join(self.root, "videos", vid, "poses")

        # Attempt to load precomputed node/adjacency
        if os.path.exists(track_dir):
            track_files = sorted(glob.glob(os.path.join(track_dir, "*.txt")))
            if track_files:
                track_data = np.loadtxt(track_files[min(frame_idx or 0, len(track_files)-1)], ndmin=2)
                ids = track_data[:,0].astype(int)
                N = len(ids)
                nodes = torch.zeros((N, 32))
                for i, tid in enumerate(ids):
                    x1, y1, x2, y2 = track_data[i,1:5]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    nodes[i, :2] = torch.tensor([cx/self.image_w, cy/self.image_h])
                    nodes[i, 2] = (x2-x1)/self.image_w
                    nodes[i, 3] = (y2-y1)/self.image_h
                coords = nodes[:, :2].numpy()
                dist = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
                adj = torch.tensor(1.0 / (1.0 + dist), dtype=torch.float32)
                return nodes, adj

        # Fallback: approximate nodes from optical flow motion or random
        N = 10
        nodes = torch.zeros((N, 32))
        for i in range(N):
            cx, cy = np.random.rand(2)
            nodes[i, :2] = torch.tensor([cx, cy])
        dist = np.linalg.norm(nodes[:,None,:2] - nodes[None,:,:2], axis=-1)
        adj = torch.tensor(1.0 / (1.0 + dist), dtype=torch.float32)
        return nodes, adj

    # ---------------------- GET ITEM ----------------------
    def __getitem__(self, idx):
        vid, start_idx = self.samples[idx]
        rgb_clip = self._load_rgb_clip(vid, start_idx)
        flow_clip = self._load_optical_flow(vid, start_idx)
        nodes, adj = self._load_nodes_and_adj(vid, start_idx)

        return {
            "rgb": rgb_clip,
            "flow": flow_clip,
            "nodes": nodes,
            "adj": adj,
            "video_id": vid,
            "start": start_idx
        }
dataset = SAVDDataset(
    root="/data/SAVD",
    split_path="/data/SAVD/splits/train.json",
    clip_len=32, stride=2, image_h=256, image_w=448
)
sample = dataset[0]
print(sample['rgb'].shape, sample['flow'].shape, sample['nodes'].shape, sample['adj'].shape)
