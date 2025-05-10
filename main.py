import os

from models.two_d.unet import Unet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
from PIL import Image
from thop import profile, clever_format
import importlib
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchio
from torchio.transforms import ZNormalization
from tqdm import tqdm
import pandas as pd
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

source_val_dir = hp.source_val_dir
label_val_dir = hp.label_val_dir

output_dir_test = hp.output_dir_test


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--best_dice_model_file', type=str, default=hp.best_dice_model_file,
                        help='Store the best_dice_model checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    training.add_argument('--best_dice', type=int, default=hp.best_dice, help='best-dice')
    parser.add_argument('-k', "--ckpt", type=str, default=hp.ckpt, help="path to the checkpoints to resume training")
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')
    return parser


import torch

def validate(model,
             val_loader,
             criterion,
             hp,
             metric,
             device="cuda"):
    """Run one full pass on the validation set.

    Returns
    -------
    (val_loss, val_dice, val_iou, val_precision, val_recall, val_FPR, val_FNR)
    """
    model.eval()

    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "FPR": 0.0,
        "FNR": 0.0,
    }
    num_iters = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if getattr(hp, "debug", False) and i >= 1:   # 可选调试
                break

            x = batch["source"]["data"].float().to(device)
            y = batch["label"]["data"].float().to(device)

            if hp.mode == "2d":
                x = x.squeeze(4)
                y = y.squeeze(4)
                y[y != 0] = 1

            outputs = model(x)
            loss = criterion(outputs, y)

            logits = torch.sigmoid(outputs)
            labels = (logits > 0.5).float()

            dice, iou, precision, recall, FPR, FNR = metric(
                y.cpu(), labels.cpu()
            )

            totals["loss"] += loss.item()
            totals["dice"] += dice
            totals["iou"] += iou
            totals["precision"] += precision
            totals["recall"] += recall
            totals["FPR"] += FPR
            totals["FNR"] += FNR
            num_iters += 1

    # 取平均
    for k in totals:
        totals[k] /= num_iters

    return (
        totals["loss"],
        totals["dice"],
        totals["iou"],
        totals["precision"],
        totals["recall"],
        totals["FPR"],
        totals["FNR"],
    )



def train(model):
    parser = argparse.ArgumentParser(description='PyTorch Image Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_train, MedData_val
    os.makedirs(args.output_dir, exist_ok=True)

    model = torch.nn.DataParallel(model, device_ids=devicess)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.02,weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,betas=(0.9, 0.99),weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=10, min_lr=1e-6)

    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0
    best_dice = args.best_dice

    model.cuda()


    writer = SummaryWriter(args.output_dir)

    train_dataset = MedData_train(source_train_dir, label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=hp.num_workers,
                              pin_memory=True, drop_last=True)

    val_dataset = MedData_train(source_val_dir, label_val_dir)
    val_loader = DataLoader(val_dataset.queue_dataset, batch_size=args.batch, shuffle=False, num_workers=hp.num_workers,
                            pin_memory=True, drop_last=False)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    for epoch in range(1, epochs + 1):
        print("epoch:" + str(epoch))
        epoch += elapsed_epochs
        num_iters = 0
        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()

            if (hp.in_class == 1) and (hp.out_class == 1) or (hp.in_class == 3) and (hp.out_class == 1):
                x = batch['source']['data']
                y = batch['label']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = y.type(torch.FloatTensor).cuda()

            else:
                x = batch['source']['data']
                y_atery = batch['atery']['data']
                y_lung = batch['lung']['data']
                y_trachea = batch['trachea']['data']
                y_vein = batch['atery']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = torch.cat((y_atery, y_lung, y_trachea, y_vein), 1)
                y = y.type(torch.FloatTensor).cuda()

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)
                y[y != 0] = 1

            outputs = model(x)
            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0

            loss = criterion(outputs, y)

            num_iters += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            iteration += 1

            dice, iou, precision, recall, FPR, FNR = metric(y.cpu(), labels.cpu())

            writer.add_scalar('Training/Loss', loss.item(), iteration)
            writer.add_scalar('Training/FPR', FPR, iteration)
            writer.add_scalar('Training/FNR', FNR, iteration)
            writer.add_scalar('Training/dice', dice, iteration)
            writer.add_scalar('Training/iou', iou, iteration)

        scheduler.step()

        # Store latest checkpoint in each epoch
        torch.save(
            {"model": model.state_dict(), "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch},
            os.path.join(args.output_dir, args.latest_checkpoint_file)
        )

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:


            model.eval()
            (val_loss,val_dice,val_iou,val_precision,val_recall,val_FPR,val_FNR) = (
                validate(model,val_loader,criterion,hp,metric,device="cuda"))

            print(f"[Epoch {epoch}] "
                  f"loss={val_loss:.4f}, "
                  f"dice={val_dice:.4f}, "
                  f"iou={val_iou:.4f}")

            writer.add_scalar('Validation/val_loss', val_loss, epoch)
            writer.add_scalar('Validation/val_dice', val_dice, epoch)
            writer.add_scalar('Validation/val_iou', val_iou, epoch)
            writer.add_scalar('Validation/val_precision', val_precision, epoch)
            writer.add_scalar('Validation/val_recall', val_recall, epoch)
            writer.add_scalar('Validation/val_FNR', val_FNR, epoch)
            writer.add_scalar('Validation/val_FPR', val_FPR, epoch)

            # scheduler.step(val_loss)
            if val_dice > best_dice:
                print(f"Dice improved from {best_dice:.4f} to {val_dice:.4f}. Saving best model...")
                best_dice = val_dice
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, "best_dice_model.pt"),
                )

    writer.close()


def test(model):
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_dir_test, exist_ok=True)

    model = torch.nn.DataParallel(model, device_ids=devicess)

    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.best_dice_model_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.best_dice_model_file),
                      map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])

    model.cuda()

    test_dataset = MedData_test(source_test_dir, label_test_dir)
    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size

    dice_scores = []
    Pids = []

    for i, subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
            subj,
            patch_size,
            patch_overlap,
        )
        print(test_dataset.image_paths[i])
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=args.batch)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)

        model.eval()

        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):

                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                if hp.mode == '2d':
                    outputs = outputs.unsqueeze(4)
                logits = torch.sigmoid(outputs)

                labels = logits.clone()
                labels[labels > 0.5] = 1
                labels[labels <= 0.5] = 0
                aggregator.add_batch(logits, locations)
                aggregator_1.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()
        affine = subj['source']['affine']

        # dice = metric(subj['label'][torchio.DATA].to(device), output_tensor_1.to(device))

        dice = metric(subj['label'][torchio.DATA].to('cpu'), output_tensor_1.to('cpu'))
        print('校验',subj['label'][torchio.DATA].shape, output_tensor_1.shape)
        dice_scores.append(dice[0].item())
        Pid = os.path.basename(test_dataset.image_paths[i])
        Pids.append(Pid)
        print(f"Dice Score for sample {i}: {dice[0].item():.4f}")

        if (hp.out_class == 1) :

            output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
            print(output_tensor_1.numpy().shape)
            output_image.save(os.path.join(output_dir_test, str(test_dataset.image_paths[i]).split('/')[-1]))
        else:
            output_tensor = output_tensor.unsqueeze(1)
            output_tensor_1 = output_tensor_1.unsqueeze(1)

            output_image_artery_float = torchio.ScalarImage(tensor=output_tensor[0].numpy(), affine=affine)
            output_image_artery_float.save(os.path.join(output_dir_test, f"{i:04d}-result_float_artery" + hp.save_arch))

            output_image_artery_int = torchio.ScalarImage(tensor=output_tensor_1[0].numpy(), affine=affine)
            output_image_artery_int.save(os.path.join(output_dir_test, f"{i:04d}-result_int_artery" + hp.save_arch))

            output_image_lung_float = torchio.ScalarImage(tensor=output_tensor[1].numpy(), affine=affine)
            output_image_lung_float.save(os.path.join(output_dir_test, f"{i:04d}-result_float_lung" + hp.save_arch))

            output_image_lung_int = torchio.ScalarImage(tensor=output_tensor_1[1].numpy(), affine=affine)
            output_image_lung_int.save(os.path.join(output_dir_test, f"{i:04d}-result_int_lung" + hp.save_arch))

            output_image_trachea_float = torchio.ScalarImage(tensor=output_tensor[2].numpy(), affine=affine)
            output_image_trachea_float.save(
                os.path.join(output_dir_test, f"{i:04d}-result_float_trachea" + hp.save_arch))

            output_image_trachea_int = torchio.ScalarImage(tensor=output_tensor_1[2].numpy(), affine=affine)
            output_image_trachea_int.save(os.path.join(output_dir_test, f"{i:04d}-result_int_trachea" + hp.save_arch))

            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[3].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_dir_test, f"{i:04d}-result_float_vein" + hp.save_arch))

            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[3].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_dir_test, f"{i:04d}-result_int_vein" + hp.save_arch))

    df = pd.DataFrame({
        "ID": Pids,  # 样本 ID 或文件名
        "Dice Score": dice_scores  # Dice 系数
    })
    df.to_csv(os.path.join(output_dir_test, "dice_scores.csv"), index=False)
    print("Dice scores saved to dice_scores.csv")

    # 打印平均 Dice 系数
    print("Average Dice Score:", sum(dice_scores) / len(dice_scores))


def set_seed(seed):
    """
    设置随机数种子，包括 CPU 和 GPU。
    """
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保结果的可重复性
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的优化，确保结果一致


if __name__ == '__main__':
    set_seed(42)
    # model_names=['unet-dropout','Unet_DC_ED','Unet_SK_E','Unet_SK_ED','Unet_SVD_SH','Unet_SVD_beforeINC']
    # model_names=['Unet','Unet_DC_ED','Unet_SK_ED','Unet_SVD_SH','Unet_SKD','Unet_SKD_SVD']
    model_names = ['unet']

    from loss_function import DiceLoss, Binary_Loss

    # criterion =  DiceLoss().cuda()
    criterion = Binary_Loss().cuda()
    # criterion =  FocalLoss().cuda()
    # criterion = DiceFocalLoss().cuda()
    # criterion = BCEDiceLoss().cuda()
    # criterion = BCEDiceFocalLoss().cuda()

    for model_name in model_names:
        print(model_name)
        model = Unet(in_channels=3, out_channels=1).to(device)
        hp.output_dir = os.path.join('logs', model_name + str(hp.init_lr))

        if hp.train_or_test == 'train':
            train(model)
        elif hp.train_or_test == 'test':
            test(model)
