'''
USAGE:
python test.py --config_file configs/ResNet_P.yaml
'''

import os
import random

import cv2
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from tqdm import tqdm
from psutil import virtual_memory
from flags import Flags
from utils import get_network
from checkpoint import load_checkpoint
from dataset import FaceDataset
from metrics import accuracy, precision, recall

from cam import apply_cam
from utils import guarantee_numpy

def main(config_file):

    options = Flags(config_file).get()

    # Set random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    os.environ["PYTHONHASHSEED"] = str(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore  # 고정하고 싶으면 False, 아니면 True

    is_cuda = torch.cuda.is_available()
    print("--------------------------------")
    print("Running {} on device {}\nWARNING: THIS IS TEST MODE!!\n".format(options.network, options.device))

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    torch.cuda.empty_cache()
    print(
        "[+] System environments\n",
        "Device: {}\n".format(torch.cuda.get_device_name(current_device)),
        "Random seed : {}\n".format(options.seed),
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    model = get_network(options)

    checkpoint = load_checkpoint(options.test_checkpoint, cuda=is_cuda)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint["epoch"]
    train_accuracy = checkpoint["train_accuracy"]
    train_recall = checkpoint["train_recall"]
    train_precision = checkpoint["train_precision"]
    train_losses = checkpoint["train_losses"]
    valid_accuracy = checkpoint["valid_accuracy"]
    valid_recall = checkpoint["valid_recall"]
    valid_precision = checkpoint["valid_precision"]
    valid_losses = checkpoint["valid_losses"]
    learning_rates = checkpoint["lr"]
    model.to(options.device)
    model.eval()

    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Checkpoint: {}\n".format(options.test_checkpoint),
        "Model parameters: {:,}\n".format(
            sum(p.numel() for p in model.parameters()),
        ),
    )

    # summary(model, (3, 224, 224), 32)

    w = options.input_size.width
    h = options.input_size.height

    transforms_test = A.Compose([
        A.Resize(w, h),
        ToTensorV2(),
    ])

    test = pd.read_csv(options.data.test)
    test['path'] = test['path'].map(lambda x : './data' + x[12:])

    test_dataset = FaceDataset(image_label=test, transforms=transforms_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=options.data.random_split, num_workers=options.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers)

    losses = []
    acces = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(test_dataloader), leave=True):
            images = images.to(options.device, torch.float)
            targets = targets.to(options.device, torch.long)

            scores = model(images).to(options.device)
            _, preds = scores.max(dim=1)

            loss = F.cross_entropy(scores, targets)
            acc = accuracy(targets, preds, options.batch_size)
            pre = precision(targets, preds)
            rec = recall(targets, preds)

            losses.append(loss.item())
            acces.append(acc)
            precisions.append(pre)
            recalls.append(rec)

    print(
        "[+] Test result\n",
        "{:10s}: {:2.8f}\n".format('Loss', np.mean(losses)),
        "{:10s}: {:2.8f}\n".format('Accuracy', np.mean(acces)),
        "{:10s}: {:2.8f}\n".format('Precision', np.mean(precisions)),
        "{:10s}: {:2.8f}\n".format('Recall', np.mean(recalls)),
    )

    # GradCAM
    # imgidx = 5
    for imgidx in range(images.shape[0]):
        rgb_img = images[imgidx].permute(1, 2, 0)
        label = targets[imgidx]
        label = 'Real Face' if label == 1 else 'Fake Face'

        # for only last block
        if options.network in ["ResNet_P", "ResNet_NP"]:
            # only last block
            # target_layers = [model.layer4[-1]]
            # apply_cam(model, guarantee_numpy(rgb_img), target_layers, label)
            # for every blocks (layer1 ~ layer4)
            blockidx = 1
            target_layers = [model.layer1[-1]]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
            blockidx += 1
            target_layers = [model.layer2[-1]]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
            blockidx += 1
            target_layers = [model.layer3[-1]]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
            blockidx += 1
            target_layers = [model.layer4[-1]]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)

        elif options.network in ["EfficientNet_P", "EfficientNet_NP"]:
            # target_layers = [model._blocks[-1]]
            # apply_cam(model, guarantee_numpy(rgb_img), target_layers, label)
            # for every blocks (._blocks[0] ~ ._blocks[15])
            blockidx = 1
            for i in range(16):
                target_layers = [model._blocks[i]]
                apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
                blockidx += 1

        elif options.network in ["DenseNet_P", "DenseNet_NP"]:
            # only last block
            # target_layers = [model._modules['features'].denseblock4]
            # apply_cam(model, guarantee_numpy(rgb_img), target_layers, label)
            # for every blocks (denseblock1, denseblock2, denseblock3, denseblock4)
            blockidx = 1
            target_layers = [model._modules['features'].denseblock1]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
            blockidx += 1
            target_layers = [model._modules['features'].denseblock2]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
            blockidx += 1
            target_layers = [model._modules['features'].denseblock3]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
            blockidx += 1
            target_layers = [model._modules['features'].denseblock4]
            apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)

        elif options.network in ["VGGNet_P", "VGGNet_NP"]:
            blockidx = 1
            for i in [4, 9, 16, 23, 30]:  # 4, 9, 16, 23, 30
                target_layers = [model._modules['features'][i]]
                apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
                blockidx += 1

        elif options.network in ["MobileNet-V2_P", "MobileNet-V2_NP"]:
            blockidx = 1
            for i in range(19):
                target_layers = [model._modules['features'][i]]
                apply_cam(model, guarantee_numpy(rgb_img), target_layers, label, imgidx=imgidx, blockidx=blockidx)
                blockidx += 1


if __name__ == '__main__':
    fire.Fire(main)