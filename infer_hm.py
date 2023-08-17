from __future__ import print_function, division

import argparse
import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models.Loss.loss import Loss_weighted, SoftIoULoss
from utils import utils
from models import HM_net
from models.data.hm_dataloader import get_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn


# test model
def test_model(args, model, test_dataloaders):
    optimizer.zero_grad()
    test_loss = 0
    loop_val = tqdm(test_dataloaders["val"], total=len(test_dataloaders["val"]))
    for data in loop_val:
        inputs = data["input"].type(torch.FloatTensor)
        outputs_room = data["output_room"].type(torch.FloatTensor)
        outputs_wall = data["output_wall"].type(torch.FloatTensor)
        outputs_points = data["output_points"].type(torch.FloatTensor)
        outputs_junctions = data["output_junctions"].type(torch.FloatTensor)

        names = data["name"]
        if args.use_gpu:
            inputs, outputs_room, outputs_wall, outputs_points, outputs_junctions = map(
                lambda t: t.to(device),
                (inputs, outputs_room, outputs_wall, outputs_points, outputs_junctions),
            )
        else:  # no
            inputs, outputs_room, outputs_wall, outputs_points, outputs_junctions = map(
                lambda t: Variable(t),
                (inputs, outputs_room, outputs_wall, outputs_points, outputs_junctions),
            )
        with torch.no_grad():
            preds = model(inputs)
            pred_points = torch.unsqueeze(preds[-1][:, 0, :, :], 1)
            pred_vertexs = torch.unsqueeze(preds[-1][:, 1, :, :], 1)
            test_loss_total = (
                nn.MSELoss()(preds[-2], outputs_room)
                + Loss_weighted(device=device)(preds[-3], outputs_wall)
                + nn.L1Loss()(pred_points, outputs_points)
                + nn.L1Loss()(pred_vertexs, outputs_junctions)
            )
            preds_wall = preds[-2].cpu().detach().numpy() * 255
            preds_room = preds[-3].cpu().detach().numpy() * 255
            preds_points = pred_points.cpu().detach().numpy() * 255
            preds_junctions = pred_vertexs.cpu().detach().numpy() * 255
            test_loss += test_loss_total.item()
            for i in range(len(preds_wall)):
                wall = preds_wall[i]
                room = preds_room[i]
                points = preds_points[i]
                junctions = preds_junctions[i]
                p = np.zeros((3, IMSIZE * 3, IMSIZE))
                p[0, :IMSIZE, :] = np.clip(room[0, :, :], 0, 255)
                p[0, IMSIZE : 2 * IMSIZE, :] = np.clip(wall[0, :, :], 0, 255)
                p[0, -IMSIZE:, :] = np.clip(points[0, :, :], 0, 255)
                p[1, -IMSIZE:, :] = np.clip(junctions[0, :, :], 0, 255)
                name = names[i]
                cv2.imwrite(
                    os.path.join(args.save_output, name), np.transpose(p, (1, 2, 0))
                )

#train model for one epoch
def train_model(args, model, dataloaders, dataset_sizes, val_dataloaders):
    writer = SummaryWriter("./logs")
    for epoch in range(args.start_epoch, args.epoches + args.start_epoch):
        for phase in ["train", "val"]:
            epoch_loss = 0
            if phase == "train":
                model.train()  # Set model to training mode
                loop = tqdm(dataloaders["train"], total=len(dataloaders["train"]))

                for data in loop:
                    optimizer.zero_grad()
                    inputs = data["input"].type(torch.FloatTensor)
                    outputs_room = data["output_room"].type(torch.FloatTensor)
                    outputs_wall = data["output_wall"].type(torch.FloatTensor)
                    outputs_points = data["output_points"].type(torch.FloatTensor)
                    outputs_junctions = data["output_junctions"].type(torch.FloatTensor)

                    # wrap them in Variable
                    if args.use_gpu:  # right
                        (
                            inputs,
                            outputs_wall,
                            outputs_room,
                            outputs_points,
                            outputs_junctions,
                        ) = map(
                            lambda t: t.to(device),
                            (
                                inputs,
                                outputs_wall,
                                outputs_room,
                                outputs_points,
                                outputs_junctions,
                            ),
                        )
                    else:  # no
                        (
                            inputs,
                            outputs_wall,
                            outputs_room,
                            outputs_points,
                            outputs_junctions,
                        ) = map(
                            lambda t: Variable(t),
                            (
                                inputs,
                                outputs_wall,
                                outputs_room,
                                outputs_points,
                                outputs_junctions,
                            ),
                        )

                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(inputs)

                        pred_points = torch.unsqueeze(preds[-1][:, 0, :, :], 1)
                        pred_vertexs = torch.unsqueeze(preds[-1][:, 1, :, :], 1)

                        loss_total = (
                            nn.MSELoss()(preds[-2], outputs_room)
                            + Loss_weighted(device=device)(preds[-3], outputs_wall)
                            + nn.L1Loss()(pred_points, outputs_points)
                            + nn.L1Loss()(pred_vertexs, outputs_junctions)
                        )

                        epoch_loss += loss_total.item()
                        loss_total.backward()
                        optimizer.step()

                    loop.set_description(
                        "Epoch [" + str(epoch) + "/" + str(args.epoches) + "]"
                    )
                    with torch.no_grad():
                        room_loss = SoftIoULoss()(preds[-3], outputs_room)
                        wall_loss = nn.MSELoss()(preds[-2], outputs_wall)
                        point_loss = nn.L1Loss()(pred_points, outputs_points)
                    loop.set_postfix(
                        learning_rate=optimizer.param_groups[0]["lr"],
                        room_loss=room_loss.item(),
                        wall_loss=wall_loss.item(),
                        point_loss=point_loss.item(),
                        loss=loss_total.item(),
                    )
                scheduler.step()
                epoch_loss = epoch_loss / dataset_sizes["train"]
                writer.add_scalar("loss", epoch_loss, epoch)

                state = {
                    "next_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if (epoch + 1) % 20 == 0:
                    torch.save(
                        state,
                        os.path.join(
                            args.save_model_dir, "{:02d}".format(epoch + 1) + ".pth"
                        ),
                    )
            elif epoch == 0 or (epoch + 1) % 10 == 0:
                optimizer.zero_grad()
                loop_val = tqdm(
                    val_dataloaders["val"], total=len(val_dataloaders["val"])
                )
                test_loss = 0
                for data in loop_val:
                    inputs = data["input"].type(torch.FloatTensor)
                    outputs_room = data["output_room"].type(torch.FloatTensor)
                    outputs_wall = data["output_wall"].type(torch.FloatTensor)
                    outputs_points = data["output_points"].type(torch.FloatTensor)
                    outputs_junctions = data["output_junctions"].type(torch.FloatTensor)

                    names = data["name"]
                    if args.use_gpu:
                        (
                            inputs,
                            outputs_room,
                            outputs_wall,
                            outputs_points,
                            outputs_junctions,
                        ) = map(
                            lambda t: t.to(device),
                            (
                                inputs,
                                outputs_room,
                                outputs_wall,
                                outputs_points,
                                outputs_junctions,
                            ),
                        )
                    else:  # no
                        (
                            inputs,
                            outputs_room,
                            outputs_wall,
                            outputs_points,
                            outputs_junctions,
                        ) = map(
                            lambda t: Variable(t),
                            (
                                inputs,
                                outputs_room,
                                outputs_wall,
                                outputs_points,
                                outputs_junctions,
                            ),
                        )
                    with torch.no_grad():
                        preds = model(inputs)
                        pred_points = torch.unsqueeze(preds[-1][:, 0, :, :], 1)
                        pred_vertexs = torch.unsqueeze(preds[-1][:, 1, :, :], 1)
                        test_loss_total = (
                            nn.MSELoss()(preds[-2], outputs_room)
                            + Loss_weighted(device=device)(preds[-3], outputs_wall)
                            + nn.L1Loss()(pred_points, outputs_points)
                            + nn.L1Loss()(pred_vertexs, outputs_junctions)
                        )
                        preds_wall = preds[-2].cpu().detach().numpy() * 255
                        preds_room = preds[-3].cpu().detach().numpy() * 255
                        preds_points = pred_points.cpu().detach().numpy() * 255
                        preds_junctions = pred_vertexs.cpu().detach().numpy() * 255
                        test_loss += test_loss_total.item()
                        for i in range(len(preds_wall)):
                            wall = preds_wall[i]
                            room = preds_room[i]
                            points = preds_points[i]
                            junctions = preds_junctions[i]
                            p = np.zeros((3, IMSIZE * 3, IMSIZE))
                            p[0, :IMSIZE, :] = np.clip(room[0, :, :], 0, 255)
                            p[0, IMSIZE : 2 * IMSIZE, :] = np.clip(
                                wall[0, :, :], 0, 255
                            )
                            p[0, -IMSIZE:, :] = np.clip(points[0, :, :], 0, 255)
                            p[1, -IMSIZE:, :] = np.clip(junctions[0, :, :], 0, 255)
                            name = names[i]
                            cv2.imwrite(
                                os.path.join(args.val_output, name),
                                np.transpose(p, (1, 2, 0)),
                            )
                        loop_val.set_postfix(test_loss=test_loss_total.item())
                test_loss /= len(val_dataloaders["val"])
                writer.add_scalar("test_loss", test_loss, epoch)
    writer.close()
    return model


# Parse arguments
parser = argparse.ArgumentParser()

# Train or Test
parser.add_argument(
    "--is_val",
    type=bool,
    default=False,
    help="Whether to add relu at the end of each HG module",
)
# Dataset paths
parser.add_argument("--tra_img_dir", type=str, help="Train image directory")
parser.add_argument("--val_img_dir", type=str, help="validate image directory")
parser.add_argument("--test_img_dir", type=str, help="test image directory")
parser.add_argument("--save_output", type=str, help="test output image directory")
parser.add_argument("--val_output", type=str, help="val output directory")
parser.add_argument("--experiment", type=str, default="A", help="name of the training")


# Checkpoint and pretrained weights
parser.add_argument(
    "--save_model_dir", type=str, help="a directory to save checkpoint file"
)
parser.add_argument(
    "--pretrained_weights",
    type=str,
    default=None,
    help="a directory to save pretrained_weights",
)

# Eval options
parser.add_argument(
    "--batch_size", type=int, default=4, help="learning rate decay after each epoch"
)
parser.add_argument(
    "--start_epoch", type=int, default=0, help="start epoch of training"
)
parser.add_argument("--epoches", type=int, default=800, help="start epoch of training")

# Network parameters
parser.add_argument("--im_size", type=int, default=256, help="size of input image")
parser.add_argument(
    "--hg_blocks", type=int, default=3, help="Number of HG blocks to stack"
)
parser.add_argument(
    "--gray_scale",
    type=str,
    default="False",
    help="Whether to convert RGB image into gray scale during training",
)
parser.add_argument(
    "--end_relu",
    type=str,
    default="True",
    help="Whether to add relu at the end of each HG module",
)

args = parser.parse_args()


""".dir file"""
# args.tra_img_dir = "./datasets/Structure3d/train"
# args.val_img_dir = "./datasets/Structure3d/test"
# args.test_img_dir = "./datasets/Structure3d/test"
args.tra_img_dir = "./datasets/Lianjia/train"
args.val_img_dir = "./datasets/Lianjia/test"
args.test_img_dir = "./datasets/Lianjia/test"


# 模型保存路径
args.save_model_dir = "./experiment/{}/ckpt".format(args.experiment)
# 输出保存路径
args.save_output = "./experiment/{}/output".format(args.experiment)
args.val_output = "./experiment/{}/val".format(args.experiment)

utils.makedir(args.save_model_dir)
utils.makedir(args.save_output)
utils.makedir(args.val_output)


"""train args"""
args.use_gpu = torch.cuda.is_available()
IMSIZE = args.im_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.is_val:
    dataloaders, dataset_sizes = get_dataset(
        args.is_val, args.test_img_dir, args.batch_size,args.im_size
    )
else:
    dataloaders, dataset_sizes = get_dataset(
        args.is_val, args.tra_img_dir, args.batch_size,args.im_size
    )
    val_dataloaders, val_dataset_sizes = get_dataset(
        True, args.val_img_dir, args.batch_size,args.im_size
    )

# use_gpu = True
model_ft = HM_net.SeqGuidNET(args.hg_blocks, args.end_relu)
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.9)
if args.is_val:
    if args.pretrained_weights != None:
        print('loading weights from {}'.format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights)
        if "state_dict" not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint["state_dict"]
            args.start_epoch = checkpoint["next_epoch"]
            model_weights = model_ft.state_dict()
            pretrained_weights = {
                k: v for k, v in pretrained_weights.items() if k in model_weights
            }
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)
    else:
        raise Exception("invalid pretrained weights path!")


if __name__ == "__main__":
    model_ft = model_ft.to(device)
    if args.is_val:
        model_ft = test_model(args, model_ft, dataloaders)
    else:
        model_ft = train_model(
            args, model_ft, dataloaders, dataset_sizes, val_dataloaders
        )
