# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.dataset_pretrain import VATDataset
from utils.utils import AverageMeter, ToDevice, print_model_info
from models.model_pretrain import VATModel
from accelerate import Accelerator
import datetime
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def custom_collate_fn(batch):
    collated_batch = {}
    # print(f'\nbatch is: {batch}\n')
    elem_keys = batch[0].keys()
    for key in elem_keys:
        if key in ['text', 'summary']:
            collated_batch[key] = [item[key] for item in batch]
        elif key in ['Text', 'Summary']:
            input_ids_tensors = [elem[key].input_ids for elem in batch]
            input_ids_tensors = [tensor.squeeze(0) for tensor in input_ids_tensors]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            attention_mask_tensors = [elem[key].attention_mask for elem in batch]
            attention_mask_tensors = [tensor.squeeze(0) for tensor in attention_mask_tensors]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True)
            collated_batch[key] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data
    return collated_batch


def train_VAT_model(train_loader, val_loader, test_loader, model, optimizer, scheduler, args, device, task, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for sci in train_loader:
            sci = ToDevice(sci, device)
            loss = model(sci)
            accelerator.backward(loss)
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()
            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()

        loss_values["train_loss"].append(np.mean(train_loss))
        val_loss = val_VAT(val_loader, model, task, device)
        test_loss = val_VAT(test_loader, model, task, device)
        loss_values["val_loss"].append(val_loss)
        loss_values["test_loss"].append(test_loss)
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'best_loss': best_loss
            }, os.path.join(args.output_path, f"checkpoint_pretrain_best.pth"))
          
            message = f"best_loss:{best_loss} ,val_loss:{val_loss}, checkpoint_pretrain_best.pth saved"
            print(message)
            with open(args.result_save_path, 'a') as f:   # 'a' means append mode, which won't overwrite existing content
                f.write(message + "\n")
        else:
            torch.save({
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'best_loss': best_loss
            }, os.path.join(args.output_path, f"checkpoint_pretrain_last.pth"))

            message = f"best_loss:{best_loss} ,val_loss:{val_loss}, ckpt checkpoint_pretrain_last saved"
            print(message)
            with open(args.result_save_path, 'a') as f:   # 'a' means append mode, which won't overwrite existing content
                f.write(message + "\n") 
        print(loss_values)

def get_latest_checkpoint(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        return None
    all_checkpoints = [filename for filename in os.listdir(checkpoints_dir) if filename.startswith("checkpoint_")]
    if not all_checkpoints:
        return None
    latest_checkpoint = max(all_checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
    return os.path.join(checkpoints_dir, latest_checkpoint)

def val_VAT(val_loader, model, task_list, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        val_loader = tqdm(val_loader, desc="Validation")
        for i, sci in enumerate(val_loader):
            sci = ToDevice(sci, device)
            loss = model(sci)
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="../ckpts/pretrain_ckpts")
    parser.add_argument("--result_save_path", type=str, default="../assets/pretrain/pretrain_log.csv")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=3)  # 12
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--task", type=int, default='6')
    parser.add_argument("--data_path", type=str,
                        default='../data')
    parser.add_argument("--data_name", type=str,
                        default='aviate')

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    train_data_file = os.path.join(args.data_path, args.data_name, 'train.json')
    val_data_file = os.path.join(args.data_path, args.data_name, 'val.json')
    test_data_file = os.path.join(args.data_path, args.data_name, 'test.json')

    task3 = {
        'inputs_modal': ['text'],
        'outputs_modal': ['summary']
    }
    task4 = {
        'inputs_modal': ['audio'],
        'outputs_modal': ['summary']
    }
    task5 = {
        'inputs_modal': ['video'],
        'outputs_modal': ['summary']
    }
    task6 = {
        'inputs_modal': ['image'],
        'outputs_modal': ['summary']
    }
    # task = [task1, task2, task3, task4, task5, task6]
    tasks = [[task6], [task3], [task4], [task5]]
    # for task in tasks:
    if args.task == 3:
        task = [task3]
    if args.task == 6:
        task = [task6]
    if args.task == 5:
        task = [task5]
    if args.task == 4:
        task = [task4]

    print(f"task : {task}")

    latest_checkpoint = get_latest_checkpoint(args.output_path)


    if latest_checkpoint:
        print(f"Latest checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found.")


    if args.mode == "train":
        logger.info(f"mode : {args.mode} ")

        accelerator = Accelerator()
        logger.info("Loading model ......")

        model = VATModel(modal=task, device=accelerator.device, args_config=args)
        best_loss = None
        if latest_checkpoint is not None:
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict=False)

        print_model_info(model, level=2)
        logger.info("Loading model successes")

        logger.info("Loading dataset ......")
        train_dataset = VATDataset(train_data_file, split="train", task=task, args_config=args)
        val_dataset = VATDataset(val_data_file, split="val", task=task, args_config=args)
        test_dataset = VATDataset(test_data_file, split="test", task=task, args_config=args)
        logger.info("Loading dataset successes")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, 3, shuffle=False, collate_fn=custom_collate_fn,
                                 pin_memory=True)
        logger.info("Loading dataloader successes")


        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        device = accelerator.device
        model = model.to(device)
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        val_loader = accelerator.prepare_data_loader(val_loader, device_placement=True)
        test_loader = accelerator.prepare_data_loader(test_loader, device_placement=True)
        train_VAT_model(train_loader, val_loader, test_loader, model, optimizer, scheduler, args, device, task, best_loss)
