# -*- coding: utf-8 -*-
"""Dump per-video predictions to CSV so uncertainty metrics can be computed offline.

Run once per checkpoint you want to report. Produces a small CSV containing no
identifiable data -- only the video's folder path, its BDI-II label, and the
model's predicted score -- so it is safe to move off the lab machine.

    python dump_predictions.py                      # defaults below

To score the ORIGINAL AUTHORS' published checkpoint instead: their weights/best.pth is a
16-base-feature BatchNorm model with four bottleneck blocks per stage. Note that four
blocks per stage is what LAYERS=2,2,2,2 builds, because ResNet._make_layer is called with
`layers[i] * k` and k is fixed at 2 below -- so the authors' test_update.py loads this
checkpoint strictly and without error. LAYERS=4,4,4,4 would build eight blocks per stage.

    CKPT=weights/best.pth LAYERS=2,2,2,2 FEATURES=16 NORM=batchnorm \
    OUT=preds_authors.csv python dump_predictions.py

Everything is read from environment variables so this never needs editing.

One deliberate difference from test_update.py: that script averages the per-batch means
(np.mean over `predict.mean()` per batch), which weights a trailing part-full batch as
heavily as a full one. Here every clip is weighted equally. The two agree exactly whenever
a video's clip count is a multiple of BATCHSIZE and differ marginally otherwise.
"""
import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from TSSTANet.tsstanet import stanet_af, group_norm_3d
from dataloader.main_dataloader import ValDataset

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_env = os.environ.get

# --- config -----------------------------------------------------------------
# Defaults reproduce the best run (fkvltxb3, f32_multistep_ms15-30).
CKPT = _env('CKPT', 'weights/f32_multistep_ms15-30/best.pth')
OUT = _env('OUT', 'preds.csv')
DATASET = _env('DATASET', 'avec14')
LAYERS = [int(x) for x in _env('LAYERS', '2,2,2,2').split(',')]
FEATURES = int(_env('FEATURES', 32))
FRAME_LEN = int(_env('FRAME_LEN', 64))
SAMPLE_INTERVAL = int(_env('SAMPLE_INTERVAL', 3))
NORM = _env('NORM', 'groupnorm')          # 'groupnorm' or 'batchnorm'
GROUPS = int(_env('GROUP_NORM_GROUPS', 8))
BATCHSIZE = int(_env('BATCHSIZE', 2))     # clips per forward pass
SCORE_RANGE = 63
# Which CSV rows to score. Matches train_update.py's held-out slice:
# avec14 -> rows 100:200, avec13 -> rows 50:100.
START = int(_env('START', 100 if DATASET == 'avec14' else 50))
END = int(_env('END', 200 if DATASET == 'avec14' else 100))
# ----------------------------------------------------------------------------

print(f'checkpoint : {CKPT}')
print(f'model      : layers={LAYERS} features={FEATURES} norm={NORM}')

norm_layer = group_norm_3d(GROUPS) if NORM == 'groupnorm' else None
Net = stanet_af(layers=LAYERS, in_channels=3, num_classes=1,
                k=2, features=FEATURES, norm_layer=norm_layer)

state = torch.load(CKPT, weights_only=True, map_location='cpu')
missing, unexpected = Net.load_state_dict(state, strict=False)
if missing or unexpected:
    # Fail loudly rather than silently scoring a partially-initialised network.
    raise SystemExit(
        f'Checkpoint does not match the model definition.\n'
        f'  missing keys   : {len(missing)}  e.g. {missing[:3]}\n'
        f'  unexpected keys: {len(unexpected)}  e.g. {unexpected[:3]}\n'
        f'Check LAYERS ({LAYERS}), FEATURES ({FEATURES}) and NORM ({NORM}).'
    )
Net = Net.to(DEVICE).eval()

df = pd.read_csv(f'./dataset/{DATASET}.csv')
paths = df['path'].values[START:END]
labels = df['label'].values[START:END]
print(f'scoring {len(paths)} videos from {DATASET}.csv rows {START}:{END}')

data = ValDataset(img_path=paths, label_value=labels, dataset=DATASET,
                  frame_len=FRAME_LEN, img_size=224, input_channel=3,
                  sample_interval=SAMPLE_INTERVAL, transform=None)
loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2,
                    drop_last=False, pin_memory=True)

rows = []
with torch.no_grad():
    for i, (pack, label) in enumerate(loader):
        clip_preds = []
        for j in range(0, pack.size(1), BATCHSIZE):
            out = Net(pack[:, j:j + BATCHSIZE].to(DEVICE).squeeze(0))
            out = torch.relu(out).view(-1) * SCORE_RANGE
            clip_preds.extend(out.cpu().tolist())
        # Paper's protocol: mean over the video's 64-frame clips is the score.
        rows.append({
            'path': paths[i],
            'label': float(label.item()),
            'pred': float(np.mean(clip_preds)),
            'n_clips': len(clip_preds),
            'clip_std': float(np.std(clip_preds)),   # within-video spread
        })
        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{len(paths)}')

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

err = np.array([r['pred'] - r['label'] for r in rows])
print(f'\nwrote {OUT}  ({len(rows)} rows)')
print(f'  MAE  {np.abs(err).mean():.4f}')
print(f'  RMSE {np.sqrt((err ** 2).mean()):.4f}')
print('\nPaper reports MAE 6.00 / RMSE 7.75 on AVEC 2014.')
print('Send this CSV -- it contains no frames and no participant identifiers'
      ' beyond the folder names already in dataset/*.csv.')
