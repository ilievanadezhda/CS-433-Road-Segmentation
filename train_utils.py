import numpy as np 
import yaml
import argparse
from omegaconf import OmegaConf
import wandb
import torch
import torchvision.transforms.v2 as transforms
from models.UNetV1 import UNetV1
from models.UNetV2 import UNetV2
from datasets.BaseDataset import BaseDataset
from datasets.TransformDataset import TransformDataset

def prepare_transforms(args):
    # prepare random transform (to be applied at the same time to both image and groundtruth, for consistency)
    # the goal is to avoid scenarios like flipping the image but not the groundtruth.
    random_transform = []
    # random resized crop
    if args.random_resized_crop:
        print(f"Using RandomResizedCrop with output size={tuple(args.output_size)}, scale={tuple(args.random_resized_crop_scale)}.")
        random_transform.append(transforms.RandomResizedCrop(size=tuple(args.output_size), scale=tuple(args.random_resized_crop_scale)))
    # random horizontal flip
    if args.random_horizontal_flip:
        print("Using RandomHorizontalFlip.")
        random_transform.append(transforms.RandomHorizontalFlip())
    # random vertical flip
    if args.random_vertical_flip:
        print("Using RandomVerticalFlip.")
        random_transform.append(transforms.RandomVerticalFlip())
    # random rotation
    if args.random_rotation:
        print(f"Using RandomRotation with degrees={args.degrees}.")
        random_transform.append(transforms.RandomRotation(degrees=args.degrees))
    # prepare additional transforms for image and groundtruth (these do not need to be applied at the same time)
    # the goal is to allow some flexibility in the transforms applied to the image and groundtruth, e.g. color jitter for image only.
    image_transform = []
    gt_transform = []
    # color jitter (for image only)
    if args.color_jitter:
        print(f"Using ColorJitter with brightness={args.brightness}, contrast={args.contrast}, saturation={args.saturation}, hue={args.hue}.")
        image_transform.append(transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue))
    # convert to tensors
    print("Using ToTensor.")
    image_transform.append(transforms.ToTensor())
    gt_transform.append(transforms.ToTensor())
    # normalization
    if args.normalization:
        print("Using Normalize with mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5).")
        image_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # gt_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))) -> No normalization for groundtruth
    print("random_transform: ", random_transform)
    print("image_transform: ", image_transform)
    print("gt_transform: ", gt_transform)
    # compose transforms
    random_transform = transforms.Compose(random_transform)
    image_transform = transforms.Compose(image_transform)
    gt_transform = transforms.Compose(gt_transform)
    # return (random_transform, image_transform, gt_transform)
    return random_transform, image_transform, gt_transform


def prepare_data(args):
    # get image and groundtruth transforms (for train set)
    random_transform, image_transform, gt_transform = prepare_transforms(args)
    # create transforms for images and groundtruths for validation and test sets
    tt_transform_image = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # TODO: Should we always normalize?
    tt_transform_gt = transforms.Compose([transforms.ToTensor()])  # No normalization for groundtruth!
    # create base dataset
    dataset = BaseDataset(image_folder=args.image_folder, gt_folder=args.gt_folder)
    # seed for reproducibility
    torch.manual_seed(args.seed)
    # split the dataset into train, validation and test set
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [int(args.train_size*len(dataset)), int(args.val_size*len(dataset)), int(args.test_size*len(dataset))])
    # apply transforms
    train_set = TransformDataset(train_set, random_transform=random_transform, image_transform=image_transform, gt_transform=gt_transform)
    val_set = TransformDataset(val_set, random_transform=None, image_transform=tt_transform_image, gt_transform=tt_transform_gt)
    test_set = TransformDataset(test_set, random_transform=None, image_transform=tt_transform_image, gt_transform=tt_transform_gt)
    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    # return data loaders
    return train_loader, val_loader, test_loader


def prepare_model(args):
    if args.model_name == "UNetV1":
        print(f"Initializing UNetV1 model with pretrained={args.model_pretrained}, scale={args.model_scale}.")
        model = UNetV1(pretrained=args.model_pretrained, scale=args.model_scale)
    elif args.model_name == "UNetV2":
        print(f"Initializing UNetV2 model with in_channels={args.model_in_channels}, out_channels={args.model_out_channels}, init_features={args.model_init_features}, pretrained={args.model_pretrained}.")
        model = UNetV2(in_channels=args.model_in_channels, out_channels=args.model_out_channels, init_features=args.model_init_features, pretrained=args.model_pretrained)
    elif args.model_name == "ResNet101":
        # TODO: Implement ResNet101 model here
        model = None  # Replace this line with the actual ResNet101 model initialization
    return model


def prepare_optimizer(model, args):
    if args.optim_name == "sgd":
        print(f"Initializing SGD optimizer with lr={args.optim_lr}, momentum={args.optim_momentum}.")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.optim_momentum)
    elif args.optim_name == "adam":
        print(f"Initializing Adam optimizer with lr={args.optim_lr}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr)
    return optimizer
  
def step_loader(loader, n_steps=-1):
    # creates infinite loader for training (
    # use as metric n_steps instead of n_epochs
    step = 0
    while True:
        for batch in loader:
            yield step, batch
            step += 1
            if step == n_steps:
                return

def train(model, device, train_loader, val_loader, criterion, optimizer, args):
    # set up WandB for logging
    wandb.init(project=args.wandb_project, name=args.wandb_run)
    # training loop
    for step, batch in step_loader(train_loader, args.n_steps):
        # training
        model.train()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # logging
        wandb.log({"Training Loss": loss.item()}, step=step)
        # validation
        if step % args.eval_freq == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            wandb.log({"Validation Loss": avg_val_loss}, step=step)
     # save model        
    torch.save(model.state_dict(), args.model_save_name) 
    # return model
    return model  
