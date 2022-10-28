import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import data_preprocess
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import CNNEncoder, RNNDecoder
from lr_scheduler import CosineAnnealingWarmup


def train_main():

    print('Load Training Data ...')
    path2data = "./train"
    class_num = os.listdir(path2data)
    video_paths = []
    labels = []

    for i in range(len(class_num)):
        Catgs = os.path.join(path2data, str(i))
        allFileList = os.listdir(Catgs)
        for name in allFileList:
            vedio = os.path.join(Catgs, name)
            video_paths.append(vedio)
            labels.append(i)

    """ DATASET parameter setting """
    IMG_SIZE = 112
    MAX_FRAME_LENGTH = 10
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 32
    
    #Data augmentation
    train_transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE,scale=(0.5,1.0)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        c
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize(mean, std),
    ])
    
    train_aug = train_transformer
    dataset = data_preprocess.VideoDataset(video_paths, labels, MAX_FRAME_LENGTH, IMG_SIZE, train_aug)
    
    # Differentiate training/validation data
    TOTAL_SIZE = len(video_paths)
    ratio = 0.8
    train_len = round(TOTAL_SIZE * ratio)
    val_len = round(TOTAL_SIZE * (1 - ratio))

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_data_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=4)
    

    model = nn.Sequential(
        CNNEncoder(),
        RNNDecoder()
    )
    
    
    # using gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0")
    model.to(device=device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0,1])

    
    """ TRAINING parameter setting """
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 1e-5) 
    lr_scheduler = CosineAnnealingWarmup(optimizer, first_cycle_steps=40, cycle_mult=1.0, max_lr=learning_rate, min_lr=1e-5, warmup_steps=0, gamma=0.5)
    epochs = 100

    min_val_loss = float("inf")

    print("Begin training...") 
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        train_hit = 0
        val_hit = 0

        for data, target in train_data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
           
            # loss function
            loss_fn = nn.CrossEntropyLoss(label_smoothing = 0.1)
            loss = loss_fn(output, target)

            total_train_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
           
            train_hit += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            # do back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        lr_scheduler.step()

        with torch.no_grad():
            model.eval()
            for data, target in val_data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_val_loss += F.cross_entropy(output, target).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                val_hit += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        avg_train_loss = total_train_loss / len(train_data_loader)
        avg_val_loss = total_val_loss / len(val_data_loader)

        print('Epoch:%3d' % epoch
              , '|Train Loss:%8.4f' % (avg_train_loss)
              , '|Train Acc:%3.4f' % (train_hit / len(train_data_loader.dataset) * 100.0)
              , '|Val Loss:%8.4f' % (avg_val_loss)
              , '|Val Acc:%3.4f' % (val_hit / len(val_data_loader.dataset) * 100.0))

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            print("-------------saving model--------------")
            # save the model (include architecture, parameters)
            torch.save(model, "model.pth")

if __name__ == "__main__":
    train_main()

