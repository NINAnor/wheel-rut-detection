import os
import glob
import numpy as np
import csv
import sklearn
import datetime
import copy
import collections

from osgeo import gdal

import torch
import torchvision

import track_zero_to_one_dataset as track_dataset
import utils
from train_modified import train_one_epoch, evaluate

train_datatype = torch.float16
train_datatype_str = 'float16'
scaler = torch.cuda.amp.GradScaler()

# train_datatype = torch.float32
# train_datatype_str = 'float32'
# scaler = None


base_dir = 'your_dir'

model_save_dir = os.path.join(base_dir, 'model_save_dir')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

base_data_dir = 'your_data_dir'
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')
test_dir = os.path.join(base_data_dir, 'test')

device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')


label_list = [0, 1]
num_classes = len(label_list)
in_channels = 3

# found in 20211210_interactive_test_sem_seg_tracks_aerial_300_rgb.py
class_weights = torch.tensor([0.5093, 27.4962])

current_class_weights = class_weights.to(device)

# NOTE: The collate function above (from vision/references/segmentation)
#       pads masks with 255. We have to add ignore_index=255 below.

def local_criterion(inputs, target):
    local_class_weights = current_class_weights
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(
            x, target.long(), weight=local_class_weights, ignore_index=255)
        
    if len(losses) == 1:
        return losses['out']
    
    return losses['out'] + 0.5 * losses['aux']

model = torchvision.models.segmentation.deeplabv3_resnet101(
    pretrained=False,
    pretrained_backbone=True,
    num_classes=num_classes)

# def append_dropout(model, dropout_probability=0.5):
#     for name, module in model.named_children():
#         if len(list(module.children())) > 0:
#             append_dropout(module, dropout_probability=dropout_probability)
#         if isinstance(module, torch.nn.Conv2d):
#             new = torch.nn.Sequential(
#                 module, torch.nn.Dropout2d(p=dropout_probability, inplace=True))
#             setattr(model, name, new)

# append_dropout(model, dropout_probability=0.3)

# def remove_double_dropout(model):
#     previous_layer = None
#     previous_name = None
#     # print(previous_layer, flush=True)
#     for name, module in model.named_children():
#         if len(list(module.children())) > 0:
#             previous_layer, previous_name = remove_double_dropout(module)
            
#         if (isinstance(module, torch.nn.Dropout2d)
#             & isinstance(previous_layer, torch.nn.Dropout2d)):
#             print('double dropout:')
            
#         previous_layer = module
#         previous_name = name
        
#     print(previous_name, flush=True)
#     return previous_layer, previous_name

# remove_double_dropout(model)

# move model to the right device
model.to(device)

max_1d_shape=600
batch_size_train = 20
num_workers_train = 20
batch_size_val = 4
num_workers_val = 4

dataset_train = track_dataset.TrackDataset(
    train_dir, train_datatype=train_datatype,
    train=True, max_1d_shape=max_1d_shape)
dataset_val = track_dataset.TrackDataset(
    val_dir, train_datatype=train_datatype,
    train=False, max_1d_shape=max_1d_shape)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, shuffle=True,
    batch_size=batch_size_train, num_workers=num_workers_train,
    collate_fn=utils.collate_fn, drop_last=True)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val, shuffle=True,
    batch_size=batch_size_val, num_workers=num_workers_val,
    collate_fn=utils.collate_fn, drop_last=True)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# optimizer = torch.optim.SGD(params, lr=0.0002,
#                             momentum=0.9, weight_decay=0.0005)
# # and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

max_lr = 0.001
min_lr = 0.0
optimizer = torch.optim.Adam(params, max_lr)

num_epochs = 8000

# scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lambda x: (1 - (x / ((len(data_loader_train) * num_epochs)
#                          + (2 * batch_size_train))) ** 0.9))

T_0 = int(np.ceil(num_epochs / 20))

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)

# # will be in cuda
# best_state_dict = copy.deepcopy(model.state_dict())

best_state_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
best_state_dict = collections.OrderedDict(best_state_dict)
best_validation_miou = 0

loss_history = np.empty((0,), dtype=float)
miou_history = np.empty((0,), dtype=float)
lr_history = np.empty((0,), dtype=float)
iou_history = np.empty((0, num_classes), dtype=float)
confusion_matrix_history = np.empty((0, num_classes, num_classes), dtype=int)

start_time = datetime.datetime.now().isoformat()
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    batch_metric_logger = train_one_epoch(model,
                                          local_criterion,
                                          optimizer, data_loader_train,
                                          lr_scheduler,
                                          device, epoch,
                                          print_freq=10,
                                          scaler=scaler,
                                          train_datatype_str=train_datatype_str)
    loss_history = np.hstack(
        [loss_history, batch_metric_logger.meters['loss'].deque])
    lr_history = np.hstack(
        [lr_history, batch_metric_logger.meters['lr'].deque])
    
    # evaluate on the test dataset
    confusion_matrix = evaluate(model, data_loader_val, device, num_classes)
    print(confusion_matrix)
    
    confusion_matrix_history = np.vstack(
        [confusion_matrix_history,
         np.expand_dims(confusion_matrix.mat.cpu().detach().numpy(), axis=0)])
    iou_tensor = confusion_matrix.compute()[2].cpu()
    miou_history = np.hstack([miou_history, iou_tensor.mean()])
    iou_history = np.vstack(
        [iou_history, confusion_matrix.compute()[2].cpu()])
    
    no_bg_miou = iou_tensor[1:].mean()
    if (no_bg_miou > best_validation_miou):
        best_state_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
        best_state_dict = collections.OrderedDict(best_state_dict)
        best_validation_miou = no_bg_miou
    
    # # if (((epoch + 1) % 10) == 0):
    # #     current_class_weights = (
    # #         class_weights / (iou_tensor + iou_tensor.mean())).to(device)
    
    # print('weights before:', current_class_weights)
    # current_class_weights = current_class_weights \
    #                         * (1 + iou_tensor.mean() - iou_tensor).to(device)
    # print('weights after:', current_class_weights)


end_time = datetime.datetime.now().isoformat()

datetime.datetime.fromisoformat(end_time) \
    - datetime.datetime.fromisoformat(start_time)


date_str = '20220222'

torch.save(model.state_dict(), os.path.join(
    model_save_dir,
    date_str + '_last_trained_model_state_dict_' + str(num_epochs)
    + '_epochs.pth'))

torch.save(best_state_dict, os.path.join(
    model_save_dir,
    date_str + '_best_iou_trained_model_state_dict_' + str(num_epochs)
    + '_epochs.pth'))

np.save(
    os.path.join(model_save_dir,
                 date_str + '_loss_history_' + str(num_epochs) + '_epochs.npy'),
    loss_history)
np.save(
    os.path.join(
        model_save_dir,
        date_str + '_lr_history_' + str(num_epochs) + '_epochs.npy'),
    lr_history)

np.save(
    os.path.join(
        model_save_dir,
        date_str + '_confusion_matrix_history_' + str(num_epochs)
        + '_epochs.npy'),
    confusion_matrix_history)
np.save(
    os.path.join(
        model_save_dir,
        date_str + '_miou_history_' + str(num_epochs) + '_epochs.npy'),
    miou_history)
np.save(
    os.path.join(
        model_save_dir,
        date_str + '_iou_history_' + str(num_epochs) + '_epochs.npy'),
    iou_history)


