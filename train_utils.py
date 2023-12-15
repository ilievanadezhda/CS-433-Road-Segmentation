import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
import torchvision.transforms.v2 as transforms
from models.UNetV1 import UNetV1
from models.UNetV2 import UNetV2
from models.UNetV3 import UNetV3
from models.DeepLabV3 import ResNet50
from datasets.BaseDataset import BaseDataset
from datasets.TransformDataset import TransformDataset
from torch.utils.data import WeightedRandomSampler
from utils import step_loader, calculate_metrics


def prepare_transforms(args):
    """ Prepare transforms for image and groundtruth.

    Args:
        args : arguments from config dictionary

    Returns:
        random_transform : random transform to be applied at the same time to both image and groundtruth
        image_transform : transform to be applied to image only
        gt_transform : transform to be applied to groundtruth only
    """
    # prepare random transform (to be applied at the same time to both image and groundtruth)
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
    # prepare image and groundtruth transforms
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
        mean, std = prepare_normalization(args.normalization_flag)
        print(f"Using Normalize with mean={mean} and std={std}.")
        image_transform.append(transforms.Normalize(mean=mean, std=std))
    # compose transforms
    if random_transform == []:
        # if there is no random transforms to be applied, set it to None
        random_transform = None
    else:
        random_transform = transforms.Compose(random_transform)
    image_transform = transforms.Compose(image_transform)
    gt_transform = transforms.Compose(gt_transform)
    # return (random_transform, image_transform, gt_transform)
    return random_transform, image_transform, gt_transform


def prepare_normalization(normalization_flag):
    """ Prepare normalization parameters.

    Args:
        normalization_flag: normalization flag to indicate which datasets are used. 
            "A": AIcrowd dataset only
            "AM": AIcrowd + Massachusetts dataset
            "AK": AIcrowd + Kaggle dataset

    Returns:
        mean: mean for normalization
        std: standard deviation for normalization
    """
    if normalization_flag == "A":
        # AIcrowd dataset only
        mean = [0.3353, 0.3328, 0.2984]
        std = [0.1967, 0.1896, 0.1897]
    elif normalization_flag == "AM":
        # AIcrowd + Massachusetts dataset
        mean = [0.3580, 0.3650, 0.3316]
        std = [0.1976, 0.1917, 0.1940]
    elif normalization_flag == "AK":
        # AIcrowd + Kaggle dataset
        mean = [0.5268, 0.5174, 0.4892]
        std = [0.1967, 0.1894, 0.1867]
    return mean, std


def prepare_sampler():
    """ Prepare weighted random sampler for training. 

    Returns:
        sampler: sampler for training
    """
    # number of samples in each dataset
    counts = {"satImage": 80, "massachusetts_384": 1333}
    # weights for each dataset
    weights = {"satImage": 1 / 80, "massachusetts_384": 1 / 1333}
    # create samples weight array
    samples_weight = np.array(
        [weights["massachusetts_384"]] * counts["massachusetts_384"]
        + [weights["satImage"]] * counts["satImage"]
    )
    # convert to tensor
    samples_weight = torch.from_numpy(samples_weight)
    # create sampler
    sampler = WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight)
    )
    # return sampler
    return sampler


def prepare_data(args):
    """ Prepare data loaders for training and validation.

    Args:
        args : arguments from config dictionary

    Returns:
        train_loader : data loader for training set
        val_loader : data loader for validation set
    """
    # get image and groundtruth transforms (for train set)
    random_transform, image_transform, gt_transform = prepare_transforms(args)
    # create image transform for validation set
    if args.normalization:
        mean, std = prepare_normalization(args.normalization_flag)
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
    train_set = BaseDataset(
        image_folders=args.train_image_folders, gt_folders=args.train_gt_folders
    )
    # load validation set
    val_set = BaseDataset(
        image_folders=args.val_image_folders, gt_folders=args.val_gt_folders
    )
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
    if len(args.train_image_folders) > 1 and args.weighted_random_sampler:
        # use sampler
        print("Using WeightedRandomSampler.")
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler = prepare_sampler(),
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True
    )
    # return data loaders
    return train_loader, val_loader


def prepare_model(args):
    """ Prepare model for training.

    Args:
        args : arguments from config dictionary

    Returns:
        model : model for training
    """
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
    elif args.model_name == "ResNet50":
        print("Initializing ResNet50 model.")
        model = ResNet50()
    elif args.model_name == "UNetV3":
        print("Initializing UNetV3 model.")
        model = UNetV3()

    return model


def prepare_optimizer(model, args):
    """ Prepare optimizer for training.

    Args:
        model : model for training
        args : arguments from config dictionary

    Returns:
        optimizer : optimizer for training
    """
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


def train(model, device, train_loader, val_loader, criterion, optimizer, args):
    """ Training loop.

    Args:
        model: model for training
        device: device to use
        train_loader: data loader for training set
        val_loader: data loader for validation set
        criterion: loss function
        optimizer: optimizer for training
        args: arguments from config dictionary

    Returns:
        model: trained model
    """
    # set up WandB for logging
    config_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=config_dict,
        entity=args.entity,
    )
    # upload the configuration file to WandB
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
            # save the model if this is the best F1 score so far
            if avg_iou > best_iou_score:
                best_iou_score = avg_iou
                torch.save(model.state_dict(), args.model_save_name)
                print("Best model saved at step: ", step)

    return model
