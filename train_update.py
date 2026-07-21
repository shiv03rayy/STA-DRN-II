# -*- coding: utf-8 -*-
import os
import csv
import time
import wandb
import torch.nn as nn
import numpy as np
import pandas as pd
from TSSTANet.tsstanet import stanet_af, group_norm_3d
import torch.optim as optim
import torch.utils.data
from torch.amp import autocast, GradScaler
from dataloader.main_dataloader import MainDataset as Dataset
from dataloader.main_dataloader import ValDataset
from dataloader import transforms3d
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def distributed_label(true_labels, classes, sigma=0.0):
    with torch.no_grad():
        true_labels = true_labels.unsqueeze(dim=1)
        true_dist = torch.arange(0, classes, 1).repeat(true_labels.size(0), 1)
        true_dist = torch.true_divide(np.exp(torch.true_divide(-(true_dist - true_labels) ** 2,
                                                               (2 * sigma ** 2))),
                                      (sigma * np.sqrt(2 * np.pi)))
    return true_dist

torch.backends.cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce allocator fragmentation on the 8GB GPU
DEVICE = torch.device('cuda')
BATCHSIZE = 1  # clips per forward pass; 8GB caps us here, effective batch recovered via BACKPROP_STEP.
               # Note this makes BatchNorm unusable -- see norm_layer below.
EPOCHS = 500
BACKPROP_STEP = 50  # effective batch = BATCHSIZE * BACKPROP_STEP ~= 50
VAL_STEP = 10
EARLY_STOP_PATIENCE = 5  # early stopping: stop after this many consecutive validations with no val-MAE improvement (each validation = VAL_STEP epochs, so 5 -> 50 epochs of no progress)
SCORE_RANGE = 63
PRETRAIN = False
DATASET = 'avec14'
SAMPLE_INTERVAL = 3
optimizer_name = 'Adam'
lr = 0.0001
LR_T_MAX = 100  # cosine anneals lr -> ~0 over this many epochs; set it to the expected run length, not EPOCHS
                # (runs so far early-stopped at 90 and 110). Past LR_T_MAX the cosine would climb back toward
                # lr, so the scheduler is frozen there -- see the stepping guard in the training loop.
LR_ETA_MIN = 1e-6  # floor, so a run that outlives LR_T_MAX holds a tiny lr rather than a hard 0 (frozen model)
frame_len = 64
features = 32  # 64 OOMs the 8GB GPU
sigma = 0
norm = 'groupnorm'

TAG = 'avec_features32_gn'


if not os.path.exists(f'weights/{TAG}'):
    os.makedirs(f'weights/{TAG}')

wandb.init(project='STA-DRN-II', name=TAG, config={
    'dataset': DATASET,
    'epochs': EPOCHS,
    'batch_size': BATCHSIZE,
    'backprop_step': BACKPROP_STEP,
    'val_step': VAL_STEP,
    'early_stop_patience': EARLY_STOP_PATIENCE,
    'optimizer': optimizer_name,
    'lr': lr,
    'lr_schedule': 'CosineAnnealingLR',
    'lr_t_max': LR_T_MAX,
    'lr_eta_min': LR_ETA_MIN,
    'frame_len': frame_len,
    'features': features,
    'sigma': sigma,
    'norm': norm,
    'sample_interval': SAMPLE_INTERVAL,
})

# Generate the model.
# GroupNorm rather than the default BatchNorm3d: at BATCHSIZE=1 the BN running stats are
# estimated from single clips, so eval() predictions diverged wildly from train().
Net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features,
                norm_layer=group_norm_3d(8))
# Net = torch.nn.DataParallel(Net)
Net = Net.to(DEVICE)
if PRETRAIN:
    Net.load_state_dict(torch.load('weights/avec_all_train/100.pth', weights_only=True, map_location=DEVICE))

# Generate the optimizers.
optimizer = getattr(optim, optimizer_name)(Net.parameters(), lr=lr)
# LR schedule: smooth cosine decay from lr to ~0 over LR_T_MAX epochs.
# Replaces MultiStepLR(milestones=[20, 40]), which cut the lr 10x at epoch 20 and again at 40 while val MAE
# was still improving at every validation (15.99 -> 10.46 -> 9.61 -> 9.24 at epochs 10/20/30/40); val MAE then
# drifted upward for the rest of the run at lr=1e-6. The assumed epoch-20 plateau was not in the data.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_T_MAX, eta_min=LR_ETA_MIN)
scaler = GradScaler()
optimizer.zero_grad()

# loss function
MSE_loss_func = nn.MSELoss()
MAE_loss_func = nn.L1Loss()
Huber_loss_func = nn.SmoothL1Loss()

# Get the dataset.
df = pd.read_csv(f'./dataset/{DATASET}.csv')
image_path_list = df['path'].values
label_list = df['label'].values

if DATASET == 'avec14':
    train_size = 100
else:
    train_size = 50
train_image_path_list = image_path_list[:train_size]
train_label_list = label_list[:train_size]
val_image_path_list = image_path_list[train_size:2 * train_size]
val_label_list = label_list[train_size:2 * train_size]


train_transform = transforms.Compose([
    transforms3d.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms3d.RandomHorizontalFlip()
])
val_transform = None

train_data = Dataset(img_path=train_image_path_list, label_value=train_label_list, dataset=DATASET,
                     frame_len=frame_len, img_size=224, input_channel=3,
                     sample_interval=SAMPLE_INTERVAL, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=10,
                          drop_last=True, pin_memory=True, persistent_workers=True)
val_data = ValDataset(img_path=val_image_path_list, label_value=val_label_list, dataset=DATASET,
                      frame_len=frame_len, img_size=224, input_channel=3,
                      sample_interval=SAMPLE_INTERVAL, transform=val_transform)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=10,
                        drop_last=False, pin_memory=True)

best_MAE = float('inf')
epochs_no_improve = 0  # early stopping: counts consecutive validations without val-MAE improvement
step_flag = 0
early_stopped = False
# Training of the model.
for epoch in range(EPOCHS):
    Net.train()
    RMSE_loss = []
    MAE_loss = []
    for step, (train_img, train_label) in enumerate(train_loader):
        step_flag += 1
        train_img = train_img.to(DEVICE)
        train_label = train_label + np.random.normal(0, sigma, train_label.shape[0])
        train_label = train_label.float().to(DEVICE) / SCORE_RANGE
        with autocast('cuda'):
            predict = Net(train_img)
            predict = predict.view(predict.size(0))
            loss = (MSE_loss_func(predict, train_label) + Huber_loss_func(predict, train_label)) / BACKPROP_STEP

        scaler.scale(loss).backward()
        if step_flag % BACKPROP_STEP == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        RMSE_loss.append(MSE_loss_func(predict * SCORE_RANGE, train_label * SCORE_RANGE).item())
        MAE_loss.append(MAE_loss_func(predict * SCORE_RANGE, train_label * SCORE_RANGE).item())
        mean_mae_loss = np.mean(MAE_loss)
        mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))

    if scheduler.last_epoch < LR_T_MAX:  # past T_max the cosine turns back upward; hold the floor instead
        scheduler.step()

    print('Epoch: {:d}  Step: {:d} | '
          'train MAE loss: {:.4f}  RMSE loss: {:.4f} | LR: {:.6f}'.format(
        epoch+1, (train_size // BATCHSIZE),
        mean_mae_loss, mean_rmse_loss, optimizer.param_groups[0]['lr']))
    wandb.log({
        'train/mae': mean_mae_loss,
        'train/rmse': mean_rmse_loss,
        'lr': optimizer.param_groups[0]['lr'],
        'epoch': epoch + 1,
    }, step=epoch + 1)

    if (epoch + 1) % VAL_STEP == 0:
        Net.eval()
        RMSE_loss = []
        MAE_loss = []
        with torch.no_grad():
            for step, (val_img_pack, val_label) in enumerate(val_loader):
                predict_list = []
                for val_img_idx in range(0, val_img_pack.size(1), BATCHSIZE):
                    with autocast('cuda'):  # match the training forward pass
                        predict = Net(val_img_pack[:, val_img_idx:val_img_idx + BATCHSIZE, :, :, :].to(DEVICE).squeeze(0))
                    predict = torch.relu(predict.float()) * SCORE_RANGE
                    predict = predict.view(predict.size(0))
                    predict_list.append(predict.mean().cpu())
                predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
                RMSE_loss.append(MSE_loss_func(predict, val_label))
                MAE_loss.append(MAE_loss_func(predict, val_label))
                # if (step + 1) % 10 == 0:
                #     print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                #         step + 1, val_label.squeeze(), predict.squeeze()))

            mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
            mean_mae_loss = np.mean(MAE_loss)
            timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
            print('{} val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))

        torch.save(Net.state_dict(), f'./weights/{TAG}/{epoch + 1}.pth')
        wandb.log({
            'val/mae': mean_mae_loss,
            'val/rmse': mean_rmse_loss,
            'epoch': epoch + 1,
        }, step=epoch + 1)
        if mean_mae_loss < best_MAE:
            best_MAE = mean_mae_loss
            epochs_no_improve = 0  # early stopping: improvement -> reset the patience counter
            torch.save(Net.state_dict(), f'./weights/{TAG}/best.pth')  # early stopping: keep the best checkpoint under a stable name
            print('Best MAE: {:.4f}, model saved!'.format(best_MAE))
        else:
            epochs_no_improve += 1  # early stopping: no improvement this validation
            print('No val-MAE improvement for {:d}/{:d} validation(s)'.format(epochs_no_improve, EARLY_STOP_PATIENCE))

        # early stopping: patience exhausted -> stop training (best.pth already holds the best weights)
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print('Early stopping at epoch {:d} | best val MAE: {:.4f}'.format(epoch + 1, best_MAE))
            early_stopped = True
            break

# Log the final result of this run so multiple runs/tags can be compared later.
results_log_path = 'results_log.csv'
log_row = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    'tag': TAG,
    'dataset': DATASET,
    'final_epoch': epoch + 1,
    'early_stopped': early_stopped,
    'best_val_mae': round(best_MAE, 4),
    'optimizer': optimizer_name,
    'lr': lr,
    'batch_size': BATCHSIZE,
    'frame_len': frame_len,
    'features': features,
    'sigma': sigma,
    'norm': norm,
}
write_header = not os.path.exists(results_log_path)
with open(results_log_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(log_row)
print(f'Run result appended to {results_log_path}')

wandb.summary['final_epoch'] = epoch + 1
wandb.summary['early_stopped'] = early_stopped
wandb.summary['best_val_mae'] = round(best_MAE, 4)
wandb.finish()
