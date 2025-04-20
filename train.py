import os
import shutil
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import my_ImageFolder
from losses import CombinedMarginLoss, ArcFace, CosFace, NaiveFace, PFace,SFace
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import my_CE, naive_CE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import logger
import importlib

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def save_config(source_files, target_folder):
    '''保存测试代码及其配置'''
    def save_file(source_file,target_folder):
        # 确保目标文件夹存在，如果不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        # 构建源文件的完整路径
        source_path = os.path.join(source_file)
        # 构建目标文件的完整路径
        target_path = os.path.join(target_folder, os.path.basename(source_file))
        # 复制文件
        shutil.copy(source_path, target_path)
        print(f"File '{source_file}' has been copied to '{target_folder}'.")
    with open(target_folder+"/__init__.py", 'w') as f:
        pass
    for source_file in source_files:
        save_file(source_file, target_folder)

def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    os.makedirs(cfg.output+"/mycode", exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    log = logger(cfg=cfg, start_step = 0, writer=summary_writer)
    # 保存配置文件
    save_config(["configs/"+config_file+".py","partial_fc_v2.py","losses.py"],cfg.output+"/mycode")
    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    # train_set = ImageFolder(cfg.rec, transform)
    train_set = my_ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.train().to(device)

    # margin_loss = CombinedMarginLoss(64, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2], cfg.interclass_filtering_threshold)
    #margin_loss = PFace()  # 直接对p+概率进行修改
    # margin_loss = SFace()  # 对scale超参进行处理
    # margin_loss = CosFace()
    margin_loss = NaiveFace()
    # margin_loss = ArcFace()
    
    CE_loss = my_CE(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    # CE_loss = naive_CE(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    # CE_loss = LoraFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.bottle_neck, False)
    # CE_loss = my_PFC(margin_loss, cfg.embedding_size, cfg.num
    # _classes, cfg.sample_rate, False)
    CE_loss.train().to(device)
    
    opt = torch.optim.SGD(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    # opt = torch.optim.Adam(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    # opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # lr_scheduler = MHLR(opt, cfg.total_step)
    # lr_scheduler = LinearLR(opt, start_factor=1, end_factor=0, total_iters=cfg.total_step)
    # lr_scheduler = ExponentialLR(opt, gamma=0.9999)
    lr_scheduler = CosineAnnealingLR(opt, T_max=cfg.total_step)


    global_step = 0
    amp = torch.GradScaler("cuda",growth_interval=100)
    for epoch in range(0, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            # if global_step >= 200:   # 此时大部分概率已经趋向于1了
            #     path_module = os.path.join(cfg.output, "model.pt")
            #     torch.save(backbone.state_dict(), path_module)
            #     # 保存FC层参数
            #     path_module = os.path.join(cfg.output, "fc.pt")
            #     torch.save(CE_loss.state_dict(), path_module)
            #     exit()
            local_embeddings = backbone(img.to(device))
            loss: torch.Tensor = CE_loss(local_embeddings, local_labels.to(device))

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()
            # lr_scheduler.my_step(loss.item())

            with torch.no_grad():
                log(global_step, loss.item(), epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # Log the std value to TensorBoard
                if hasattr(CE_loss, 'std'):  # Ensure that the 'std' attribute exists
                    summary_writer.add_scalar('train/std', CE_loss.std**2, global_step)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": CE_loss.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{epoch}.pt"))

        # path_module = os.path.join(cfg.output, f"model_{epoch}.pt")
        # torch.save(backbone.state_dict(), path_module)
        # path_module = os.path.join(cfg.output, "fc.pt")
        # torch.save(CE_loss.state_dict(), path_module)

    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(backbone.state_dict(), path_module)
    # 保存FC层参数
    path_module = os.path.join(cfg.output, "fc.pt")
    torch.save(CE_loss.state_dict(), path_module)
    log.loss2csv(cfg.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="mnist", help='the name of config file')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
    # 训练完成后进行准确率测试
    import eval_ijbb
    import eval_ijbc
