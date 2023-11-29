import numpy as np
import argparse
from omegaconf import OmegaConf
import wandb
import torch
import torchvision.transforms.v2 as transforms
from models.UNetV1 import UNetV1
from models.UNetV2 import UNetV2
from models.UNetV3 import UNetV3
from models.DeepLabV3 import ResNet101
from datasets.BaseDataset import BaseDataset
from datasets.TransformDataset import TransformDataset
import random


# set seeds as function
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_transforms(args):
    # prepare random transform (to be applied at the same time to both image and groundtruth, for consistency).
    # the goal is to avoid scenarios like flipping the image but not the groundtruth.
    random_transform = []
    # random resized crop
    if args.random_resized_crop:
        print(
            f"Using RandomResizedCrop with output size={tuple(args.output_size)}, scale={tuple(args.random_resized_crop_scale)}."
        )
        random_transform.append(
            transforms.RandomResizedCrop(
                size=tuple(args.output_size),
                scale=tuple(args.random_resized_crop_scale),
            )
        )
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
        print(
            f"Using ColorJitter with brightness={args.brightness}, contrast={args.contrast}, saturation={args.saturation}, hue={args.hue}."
        )
        image_transform.append(
            transforms.ColorJitter(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                hue=args.hue,
            )
        )
    # resize
    print(f"Using Resize with input size={args.input_size}.")
    image_transform.append(transforms.Resize((args.input_size, args.input_size)))
    gt_transform.append(transforms.Resize((args.input_size, args.input_size)))
    # convert to tensors
    print("Using ToTensor.")
    image_transform.append(transforms.ToTensor())
    gt_transform.append(transforms.ToTensor())
    # normalization
    if args.normalization:
        mean = [0.3580, 0.3650, 0.3316]
        std = [0.1976, 0.1917, 0.1940]
        # mean = [0.3353, 0.3328, 0.2984]
        # std = [0.1967, 0.1896, 0.1897]
        print(f"Using Normalize with mean={mean} and std={std}.")
        image_transform.append(transforms.Normalize(mean=mean, std=std))
    # compose transforms
    # if there is no random transforms to be applied, set it to None
    if random_transform == []:
        random_transform = None
    else:
        random_transform = transforms.Compose(random_transform)
    image_transform = transforms.Compose(image_transform)
    gt_transform = transforms.Compose(gt_transform)
    # return (random_transform, image_transform, gt_transform)
    return random_transform, image_transform, gt_transform


def prepare_data(args):
    # get image and groundtruth transforms (for train set)
    random_transform, image_transform, gt_transform = prepare_transforms(args)
    # create image transform for validation set
    if args.normalization:
        mean = [0.3580, 0.3650, 0.3316]
        std = [0.1976, 0.1917, 0.1940]
        # mean = [0.3353, 0.3328, 0.2984]
        # std = [0.1967, 0.1896, 0.1897]
        tt_transform_image = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        tt_transform_image = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ]
        )
    # create groundtruth transform for validation set
    tt_transform_gt = transforms.Compose(
        [transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor()]
    )
    # load train set
    train_set = BaseDataset(image_folders=args.train_image_folders, gt_folders=args.train_gt_folders)
    # load validation set
    val_set = BaseDataset(image_folders=args.val_image_folders, gt_folders=args.val_gt_folders)
    # apply transforms
    train_set = TransformDataset(
        train_set,
        random_transform=random_transform,
        image_transform=image_transform,
        gt_transform=gt_transform,
    )
    val_set = TransformDataset(
        val_set,
        random_transform=None,
        image_transform=tt_transform_image,
        gt_transform=tt_transform_gt,
    )
    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True
    )
    # return data loaders
    return train_loader, val_loader


def prepare_model(args):
    if args.model_name == "UNetV1":
        print(
            f"Initializing UNetV1 model with pretrained={args.model_pretrained}, scale={args.model_scale}."
        )
        model = UNetV1(pretrained=args.model_pretrained, scale=args.model_scale)
    elif args.model_name == "UNetV2":
        print(
            f"Initializing UNetV2 model with in_channels={args.model_in_channels}, out_channels={args.model_out_channels}, init_features={args.model_init_features}, pretrained={args.model_pretrained}."
        )
        model = UNetV2(
            in_channels=args.model_in_channels,
            out_channels=args.model_out_channels,
            init_features=args.model_init_features,
            pretrained=args.model_pretrained,
        )
    elif args.model_name == "ResNet101":
        print("Initializing ResNet101 model.")
        model = ResNet101()
    elif args.model_name == "UNetV3":
        print("Initializing UNetV3 model.")
        model = UNetV3()
    # ADD MODELS HERE!
    return model


def prepare_optimizer(model, args):
    if args.optim_name == "sgd":
        print(
            f"Initializing SGD optimizer with lr={args.optim_lr}, momentum={args.optim_momentum}."
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.optim_momentum
        )
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


def calculate_metrics(preds, labels, threshold=0.5):
    smooth = 1e-6
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    # Pixel Accuracy
    pixel_accuracy = (preds == labels).sum().item() / (labels.numel() + smooth)

    # Intersection-Over-Union (IoU)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # Precision and Recall for F1 Score
    true_positives = (preds * labels).sum()
    predicted_positives = preds.sum()
    actual_positives = labels.sum()

    precision = true_positives / (predicted_positives + smooth)
    recall = true_positives / (actual_positives + smooth)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)

    return pixel_accuracy, iou.item(), f1_score.item()


def train(model, device, train_loader, val_loader, criterion, optimizer, args):
    # set up WandB for logging
    config_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=config_dict,
        entity=args.entity,
    )
    # Upload the configuration file to WandB
    wandb.config.update(config_dict)
    best_iou_score = 0.0
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
        train_pixel_accuracy, train_iou, train_f1 = calculate_metrics(outputs, labels)
        wandb.log(
            {
                "Training Pixel Accuracy": train_pixel_accuracy,
                "Training IoU": train_iou,
                "Training F1 Score": train_f1,
            },
            step=step,
        )
        # TODO
        # validation
        if step % args.eval_freq == 0:
            model.eval()
            total_val_loss = 0.0
            total_pixel_accuracy = 0.0
            total_iou = 0.0
            total_f1 = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
                    total_val_loss += val_loss.item()

                    # calculate metrics for validation data
                    val_pixel_accuracy, val_iou, val_f1 = calculate_metrics(
                        val_outputs, val_targets
                    )
                    total_pixel_accuracy += val_pixel_accuracy
                    total_iou += val_iou
                    total_f1 += val_f1

            avg_pixel_accuracy = total_pixel_accuracy / len(val_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            avg_iou = total_iou / len(val_loader)
            avg_f1 = total_f1 / len(val_loader)
            wandb.log(
                {
                    "Average Validation Loss": avg_val_loss,
                    "Average Validation Pixel Accuracy": avg_pixel_accuracy,
                    "Average Validation IoU": avg_iou,
                    "Average Validation F1 Score": avg_f1,
                },
                step=step,
            )
            # Save the model if this is the best F1 score so far
            if avg_iou > best_iou_score:
                best_iou_score = avg_iou
                torch.save(model.state_dict(), args.model_save_name)
                print("Best model saved at step: ", step)

    return model


# calculate mean and std of dataset
def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
    return mean, std
