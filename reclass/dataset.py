import os
import json
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class UnifiedDataset(Dataset):
    """
    Unified dataset loader for RECLASS.
    Supports SAVD (multimodal), UCSD Ped1/Ped2, and Avenue datasets.
    Automatically computes optical flow and graph adjacency if not available.
    """

    def __init__(self, name, root, split="train",
                 clip_len=32, stride=2, image_h=256, image_w=448):
        self.name = name.lower()
        self.root = root
        self.split = split
        self.clip_len = clip_len
        self.stride = stride
        self.image_h = image_h
        self.image_w = image_w
        self.flow_calc = cv2.optflow.DualTVL1OpticalFlow_create()

        if self.name == "savd":
            self.video_list = self._load_savd_split()
            self.dataset_type = "savd"
        elif "ped" in self.name:
            self.video_list = sorted(glob.glob(os.path.join(root, "Train" if split=="train" else "Test", "*")))
            self.dataset_type = "ucsd"
        elif "avenue" in self.name:
            subdir = "training_videos" if split=="train" else "testing_videos"
            self.video_list = sorted(glob.glob(os.path.join(root, subdir, "*")))
            self.dataset_type = "avenue"
        else:
            raise ValueError(f"Unknown dataset {name}")

        self.samples = self._prepare_index()

    # ---------------------- SAVD SPLIT ----------------------
    def _load_savd_split(self):
        split_file = os.path.join(self.root, "splits", f"{self.split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"SAVD split file not found: {split_file}")
        with open(split_file, "r") as f:
            vids = json.load(f)
        return vids

    def _prepare_index(self):
        samples = []
        if self.dataset_type == "savd":
            for vid in self.video_list:
                rgb_dir = os.path.join(self.root, "videos", vid, "rgb")
                frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                    samples.append((vid, start))
        else:
            # UCSD / Avenue
            for video_path in self.video_list:
                frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
                for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                    samples.append((video_path, start))
        return samples

    # ---------------------- RGB CLIP ----------------------
    def _load_rgb_clip(self, frames):
        imgs = []
        for f in frames:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_w, self.image_h))
            imgs.append(torch.from_numpy(img).permute(2,0,1).float() / 255.0)
        return torch.stack(imgs, dim=0)

    # ---------------------- FLOW ----------------------
    def _load_or_compute_flow(self, frames):
        flows = []
        for i in range(len(frames)-1):
            prev = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
            nxt = cv2.imread(frames[i+1], cv2.IMREAD_GRAYSCALE)
            prev = cv2.resize(prev, (self.image_w, self.image_h))
            nxt = cv2.resize(nxt, (self.image_w, self.image_h))
            flow = self.flow_calc.calc(prev, nxt, None)
            flow = np.clip(flow, -20, 20)
            flows.append(torch.from_numpy(flow.transpose(2,0,1)).float())
        # pad to clip_len
        while len(flows) < self.clip_len:
            flows.append(torch.zeros_like(flows[-1]))
        return torch.stack(flows, dim=0)

    # ---------------------- GRAPH ----------------------
    def _load_or_compute_graph(self, vid):
        """For SAVD: load or synthesize graph. For UCSD/Avenue: return zero graph."""
        if self.dataset_type != "savd":
            N = 1
            return torch.zeros((N, 32)), torch.zeros((N, N))

        track_dir = os.path.join(self.root, "videos", vid, "tracks")
        if not os.path.exists(track_dir):
            N = 10
            nodes = torch.randn(N, 32)
            adj = torch.sigmoid(torch.randn(N, N))
            return nodes, adj

        last_track = sorted(glob.glob(os.path.join(track_dir, "*.txt")))[-1]
        data = np.loadtxt(last_track, ndmin=2)
        ids = data[:,0].astype(int)
        N = len(ids)
        nodes = torch.zeros((N, 32))
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = data[i,1:5]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            nodes[i, :2] = torch.tensor([cx/self.image_w, cy/self.image_h])
            nodes[i, 2] = (x2-x1)/self.image_w
            nodes[i, 3] = (y2-y1)/self.image_h
        coords = nodes[:, :2].numpy()
        dist = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
        adj = torch.tensor(1.0 / (1.0 + dist), dtype=torch.float32)
        return nodes, adj

    # ---------------------- GET ITEM ----------------------
    def __getitem__(self, idx):
        vid, start = self.samples[idx]
        if self.dataset_type == "savd":
            rgb_dir = os.path.join(self.root, "videos", vid, "rgb")
            frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))[start:start+self.clip_len]
        else:
            frames = sorted(glob.glob(os.path.join(vid, "*.jpg")))[start:start+self.clip_len]

        rgb_clip = self._load_rgb_clip(frames)
        flow_clip = self._load_or_compute_flow(frames)
        nodes, adj = self._load_or_compute_graph(vid if self.dataset_type=="savd" else None)

        return {
            "rgb": rgb_clip,
            "flow": flow_clip,
            "nodes": nodes,
            "adj": adj,
            "video_id": vid,
            "start": start
        }

    def __len__(self):
        return len(self.samples)
# SAVD
savd = UnifiedDataset(name="SAVD", root="/data/SAVD", split="train")

# UCSD Ped2
ped2 = UnifiedDataset(name="Ped2", root="/data/UCSD/Ped2", split="test")

# Avenue
avenue = UnifiedDataset(name="Avenue", root="/data/Avenue", split="train")

for sample in [savd[0], ped2[0], avenue[0]]:
    print(sample['video_id'], sample['rgb'].shape, sample['flow'].shape)
