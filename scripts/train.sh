#!/usr/bin/env bash
python - <<'PY'
from yaml import safe_load
from reclass.training import Trainer
from reclass.dataset import DummyDataset
cfg = safe_load(open('configs/default.yaml'))
ds = DummyDataset(cfg)
trainer = Trainer(cfg, ds)
trainer.train()
PY
