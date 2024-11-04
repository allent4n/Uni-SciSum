# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import csv
import argparse
import json
from tqdm import tqdm
import numpy as np
from rouge import rouge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score

import torch.nn.functional as F
from dataset.dataset_finetune_unimodal import gitDataset

from utils.utils import AverageMeter, ToDevice, print_model_info
from models.model_finetune_unimodal import SciSumModel
from accelerate import Accelerator
import datetime
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    for key in elem_keys:
        if key in ['src_text', 'summary', 'gold_figs']:
            collated_batch[key] = [item[key] for item in batch]
        elif key in ['src_Text', 'tgt_Caption']:
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

def transform_img_ind(list_a, list_b):
    result = []
    
    for sublist_a, label in zip(list_a, list_b): # shape then label list
        # Create a new sublist of zeros with the same length as sublist_a
        new_sublist = [0] * len(sublist_a)
        
        # Set the position of the 1 according to the value in list_b (label)
        if label < len(new_sublist):
            new_sublist[label] = 1
        else:
            new_sublist[0] = 1
        
        result.append(new_sublist)
    
    return result

def calculate_top_k_accuracy(predicted_frames, real_frames, k_values=[1, 3], padding_value=-1):
    
    # Step 1: Find the maximum length of the predicted lists
    max_len = max(len(pred) for pred in predicted_frames)
    
    # Step 2: Pad the shorter predicted lists with the padding_value
    predicted_frames_padded = [pred + [padding_value] * (max_len - len(pred)) for pred in predicted_frames]
    
    # Step 3: Define the full set of possible labels (0 to max_len-1)
    labels = list(range(max_len))
    
    # Initialize a dictionary to store Top-K accuracy results
    accuracy_results = {}
    
    # Step 4: Calculate Top-K Accuracy for each k in k_values
    for k in k_values:
        accuracy = top_k_accuracy_score(real_frames, np.array(predicted_frames_padded), k=k, labels=labels)
        accuracy_results[f"Top-{k} Accuracy"] = accuracy * 100  # Convert to percentage
    
    return accuracy_results

def get_latest_checkpoint(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        return None

    all_checkpoints = [filename for filename in os.listdir(checkpoints_dir) if filename.startswith("checkpoint_")]

    if not all_checkpoints:
        return None

    latest_checkpoint = max(all_checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
    return os.path.join(checkpoints_dir, latest_checkpoint)


def train_git_decoder(train_loader, val_loader, test_loader, model, optimizer, scheduler, args, device, task,
                      best_loss=None):
    best_rl = 0
    running_loss = AverageMeter()
    step = 0
    patience_count = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")

        # model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for sci in train_loader:
            if "summary" in sci:
                del sci["summary"]
                del sci['gold_figs']

            sci = ToDevice(sci, device)
            loss = model(sci)
            accelerator.backward(loss)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()

        if patience_count < args.patience:
            print(f"patience_count: {patience_count}")
            if (epoch + 1) % 3 == 0:
                loss_values["train_loss"].append(np.mean(train_loss))
                val_loss, rl_score = val_git(val_loader, model, task, device)
                loss_values["val_loss"].append(val_loss)
                if best_loss == None or rl_score > best_rl:
                    best_rl = rl_score
                    best_loss = val_loss
                    torch.save({
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'best_loss': best_loss
                    }, os.path.join(args.output_path, f"checkpoint_best.pth"))

                    message = f"best_loss:{best_loss} ,val_loss:{val_loss}, checkpoint_best.pth saved"
                    print(message)
                    # Write the message to the file
                    with open(args.result_save_path, 'a') as f:
                        f.write(message + "\n")
                else:
                    patience_count += 1
                    torch.save({
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'best_loss': best_loss
                    }, os.path.join(args.output_path, f"checkpoint_last.pth"))

                    message = f"best_loss:{best_loss} ,val_loss:{val_loss}, ROUGE-L: {rl_score}, Best_RL: {best_rl}, checkpoint_last.pth saved"
                    print(message)
                    with open(args.result_save_path, 'a') as f:
                        f.write(message + "\n")
                print(f'train: {loss_values["train_loss"]}')
                print(f'val: {loss_values["val_loss"]}')
        else:
            break


def val_git(val_loader, model, task_list, device):
    model.eval()
    val_loss = 0
    all_gt, all_pred = [], []
    logger.info("Validating...")
    with torch.no_grad():
        _real_frames, _predicted_frames, _tran_pred_frames = [], [], []
        _img_id_predicted = 0
        val_loader = tqdm(val_loader, desc="Validation")
        for i, sci in enumerate(val_loader):
            gold_figs = sci['gold_figs']
            if "src_text" in sci:
                src_text = sci["src_text"]
                del sci["src_text"]
                
            if "summary" in sci:
                gt_summary = sci["summary"]
                del sci["summary"]
                del sci['gold_figs']

            sci = ToDevice(sci, device)
            sci['src_text'] = src_text
            loss = model(sci)

            for task in task_list:
                print(f"task : {task}")
                inputs_modal = task['inputs_modal']
                outputs_modal = task['outputs_modal']

                if 'summary' in outputs_modal:
                    src_text, result = model.generate_text(sci, inputs_modal, outputs_modal)
                    all_gt.extend(gt_summary)
                    all_pred.extend(result)
                    print(f"gt_summary : {gt_summary[0]}")
                    print(f"pred_summary : {result[0]}")
                    
                if 'image_ind' in outputs_modal:
                    for _p_sent in result:
                        try:
                            _, _frame = _p_sent.split("img_ind_")
                            try:
                                _frame = int(_frame[-1].split(".")[0])
                            except:
                                _frame = int(_frame[-1])
                        except:
                            _frame = 4
                        else:
                            _img_id_predicted += 1
                        _predicted_frames.append(_frame)
                        
                    for _p_sent in gt_summary:
                        _, _src_frame = _p_sent.split("img_ind_")
                        _src_frame = int(_src_frame)
                        _real_frames.append(_src_frame)
                    
                    _tem_pred_tran_frames = transform_img_ind(gold_figs, _predicted_frames)
                    for i in _tem_pred_tran_frames:
                        _tran_pred_frames.append(i)

            val_loss += loss.detach().cpu().item()
        
        if 'summary' in outputs_modal:
            print('\n>>>>>>>>>>>>>> Rouge >>>>>>>>>>>>>>')
            print(f"all gt_summary : {len(all_gt)}")
            print(f"all pred_summary : {len(all_pred)}")
            rougel_score = rouge_scorer(all_pred, all_gt)
            
        if 'image_ind' in outputs_modal:
            print('\n>>>>>>>>>>>>>> Img Acc >>>>>>>>>>>>>>')
            print(f'all trans_framws: {len(_tran_pred_frames)}')
            print(calculate_top_k_accuracy(_tran_pred_frames, _real_frames))
            
        print('\n>>>>>>>>>>>>>> Val Loss >>>>>>>>>>>>>>')
        logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader), rougel_score


def test_git(test_loader, model, task_list, device):
    model.eval()
    logger.info("Testing...")
    with torch.no_grad():
        _real_frames, _predicted_frames, _tran_pred_frames = [], [], []
        _img_id_predicted = 0
        test_loader = tqdm(test_loader, desc="Test")
        for task in task_list:
            print(f"task : {task}")
            summary_list = []
            pre_summary_list = []
            src_text_list = []
            for sci in test_loader:
                gold_figs = sci['gold_figs']
                
                summary = sci['summary']
                summary_list = summary_list + summary

                if "summary" in sci:
                    del sci["summary"]
                    del sci['gold_figs']

                sci = ToDevice(sci, device)
                inputs_modal = task['inputs_modal']
                outputs_modal = task['outputs_modal']

                if 'summary' in outputs_modal:
                    src_text, pre_summary = model.generate_text(sci, inputs_modal, outputs_modal)
                    print(f"gt_summary : {summary[0]} , pred_summary : {pre_summary[0]}")
                    pre_summary_list = pre_summary_list + pre_summary
                    src_text_list = src_text_list + src_text
                    
                if 'image_ind' in outputs_modal:
                    for _p_sent in pre_summary:
                        try:
                            _frame = _p_sent.split("img_ind_")
                            try:
                                _frame = int(_frame[-1].split(".")[0])
                            except:
                                _frame = int(_frame[-1])
                        except:
                            _frame = 0
                        else:
                            _img_id_predicted += 1
                        _predicted_frames.append(_frame)
                        
                    for _p_sent in summary:
                        _, _src_frame = _p_sent.split("img_ind_")
                        _src_frame = int(_src_frame)
                        _real_frames.append(_src_frame)

                    _tem_pred_tran_frames = transform_img_ind(gold_figs, _predicted_frames)
                    for i in _tem_pred_tran_frames:
                        _tran_pred_frames.append(i)

            if 'summary' in outputs_modal:
                print('\n>>>>>>>>>>>>>> Rouge >>>>>>>>>>>>>>')
                rouge_scorer(pre_summary_list, summary_list)
                assert len(summary_list) == len(pre_summary_list), "Lists must have the same length."

                csv_file_name = '../../assets/generated_summary/summaries.csv'
                with open(csv_file_name, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['src_text', 'pred_result', 'results'])
                    for item_0, item_A, item_B in zip(src_text_list, pre_summary_list, summary_list):
                        csvwriter.writerow([item_0, item_A, item_B])
                        
            if 'image_ind' in outputs_modal:
                print('\n>>>>>>>>>>>>>> Img Acc >>>>>>>>>>>>>>')
                print(calculate_top_k_accuracy(_tran_pred_frames, _real_frames))


def rouge_scorer(hyp_list, ref_list):
    # Initialize the scorer
    scorer = rouge.rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate scores
    scores = [scorer.score(ref, hyp) for hyp, ref in zip(hyp_list, ref_list)]

    sums = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
    }

    # Sum up all the scores
    for score in scores:
        for key in sums.keys():
            sums[key]['precision'] += score[key].precision
            sums[key]['recall'] += score[key].recall
            sums[key]['fmeasure'] += score[key].fmeasure

    # Calculate averages
    num_pairs = len(hyp_list)
    averages = {
        key: {
            'precision': sums[key]['precision'] / num_pairs,
            'recall': sums[key]['recall'] / num_pairs,
            'fmeasure': sums[key]['fmeasure'] / num_pairs
        }
        for key in sums.keys()
    }

    # Output the average scores
    print("Average ROUGE scores:")
    for key in averages.keys():
        print(f"{key}:")
        for metric in ['precision', 'recall', 'fmeasure']:
            print(f"  {metric}: {averages[key][metric]:.4f}")
            
    return averages[key][metric] # ROUGE_L fmeasure


def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="../../ckpts/finetune_ckpts")
    parser.add_argument("--pretrain", type=str, default="../../ckpts/pretrain_ckpts")
    parser.add_argument("--result_save_path", type=str,
                        default="../../assets/finetune/result.csv")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--data_path", type=str,
                        default='../../data')
    parser.add_argument("--data_name", type=str, default='cellpress')
    parser.add_argument("--prompt", type=bool, default=True)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    task1 = {
        'inputs_modal': ['video'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task2 = {
        'inputs_modal': ['audio'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task3 = {
        'inputs_modal': ['image'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task4 = {
        'inputs_modal': ['src_text'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task1_1 = {
        'inputs_modal': ['src_text', 'video'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task2_1 = {
        'inputs_modal': ['src_text', 'audio'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task3_1 = {
        'inputs_modal': ['src_text', 'image'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task4_1 = {
        'inputs_modal': ['src_text', 'video', 'audio'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task5_1 = {
        'inputs_modal': ['src_text', 'video', 'image'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task6_1 = {
        'inputs_modal': ['src_text', 'audio', 'image'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task0 = {
        'inputs_modal': ['src_text', 'audio', 'video', 'image'],
        'outputs_modal': ['summary', 'image_ind']
    }
    task0_1 = {
        'inputs_modal': ['src_text', 'audio', 'video', 'image'],
        'outputs_modal': ['image_ind']
    }
    task0_2 = {
        'inputs_modal': ['src_text', 'audio', 'video', 'image'],
        'outputs_modal': ['summary']
    }

    # load dataset
    train_data_file = os.path.join(args.data_path, args.data_name, 'train.json')
    val_data_file = os.path.join(args.data_path, args.data_name, 'val.json')
    test_data_file = os.path.join(args.data_path, args.data_name, 'test.json')

    task = [task0]

    print(f"task : {task}")

    latest_checkpoint = get_latest_checkpoint(args.output_path)

    pretrain_checkpoint = get_latest_checkpoint(args.pretrain)

    if latest_checkpoint:
        print(f"Git-decoder checkpoint: {latest_checkpoint}")
    elif latest_checkpoint is None and pretrain_checkpoint:
        print(f"Git-base checkpoint: {pretrain_checkpoint}")
    else:
        print("No checkpoint found.")

    if args.mode == "train":
        logger.info(f"mode : {args.mode} ")
        accelerator = Accelerator()
        logger.info("Loading model ......")
        model = SciSumModel(task=task, device=accelerator.device, args_config=args)

        if latest_checkpoint is not None:
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict=False)

        if latest_checkpoint is None and pretrain_checkpoint is not None:
            # Filter the state_dict
            state_dict = torch.load(pretrain_checkpoint, map_location='cpu')["model_state_dict"]
            keys_to_keep = {'model.query_tokens', 'model.git_former', 'qformer'}
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in keys_to_keep}
            # best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(filtered_state_dict, strict=False)
            # state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            # best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            # model.load_state_dict(state_dict, strict=False)

        best_loss = None

        print_model_info(model, level=2)
        logger.info("Loading model successed")

        logger.info("Loading dataset ......")
        train_dataset = gitDataset(train_data_file, args_config=args, split="train", task=task)
        val_dataset = gitDataset(val_data_file, args_config=args, split="val", task=task)
        test_dataset = gitDataset(test_data_file, args_config=args, split="test", task=task)
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

        val_loader = DataLoader(val_dataset, 2, shuffle=False, collate_fn=custom_collate_fn,
                                num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, 2, shuffle=False, collate_fn=custom_collate_fn, 
                                num_workers=args.num_workers,
                                pin_memory=True)
        logger.info("Loading dataloader successed")

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        device = accelerator.device
        model = model.to(device)
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        val_loader = accelerator.prepare_data_loader(val_loader, device_placement=True)
        test_loader = accelerator.prepare_data_loader(test_loader, device_placement=True)
        train_git_decoder(train_loader, val_loader, test_loader, model, optimizer, scheduler, args, device, task,
                          best_loss)

    elif args.mode == "test":
        logger.info(f"mode : {args.mode} ")
        logger.info("Loading dataset ......")
        test_dataset = gitDataset(test_data_file, args_config=args, split="test", task=task)
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        test_loader = DataLoader(test_dataset, 4, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        logger.info("Loading model ......")
        accelerator = Accelerator()
        device = accelerator.device
        model = SciSumModel(task=task, device=device, args_config=args)
        if latest_checkpoint is not None:
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict=False)

        print_model_info(model, level=2)
        logger.info("Loading model successed")
        model = model.to(device)
        test_loader = accelerator.prepare_data_loader(test_loader, device_placement=True)
        test_git(test_loader, model, task, device)


