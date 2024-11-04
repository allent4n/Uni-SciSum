# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
import copy
from tqdm import tqdm
import json
import re
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from transformers import BertTokenizer, T5Tokenizer, AutoTokenizer
from transformers import Blip2Processor
from torchvision import transforms
import torch.nn as nn
import re
import numpy as np
import random
from itertools import combinations
from rdkit.Chem.rdchem import HybridizationType, BondType



BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

class BaseDataset(Dataset): #, ABC):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self._load_data()
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.text)




def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    """
    greedily selects top summary_size sentences
    """

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return sorted(selected)  # selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


# utilities:
def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    '''

    :param evaluated_ngrams:  list
    :param reference_ngrams:  list
    :return:
    '''
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def separate_sentences(text):
    # Use a regular expression to split the text into sentences
    sentences = re.split(r'\.\s+', text)

    # Remove the last empty string if the text ends with a period
    if sentences[-1] == '':
        sentences.pop()

    # Wrap each sentence in a list and add all lists to a main list
    sentence_lists = [[sentence.strip() + '.'] for sentence in sentences if sentence]

    return sentence_lists
def process_video_meta(data):
    results = {}
    idx = 0
    for vid, data_item in data.items():
        duration = float(data_item['duration'])
        # for timestamp, sentence, summary in zip(data_item["timestamps"], data_item["sentences"],
        #                                         data_item['sentences']):
        for timestamp, sentence, summary, figs, gold_figs in zip(data_item["timestamps"], data_item["text"],
                                                                 data_item['abstract'],
                                                                 data_item["figures"], data_item["GA"]):
            start_time = max(0.0, float(timestamp[0]))
            end_time = min(float(timestamp[1]), duration)

            sents = " ".join(sentence)
            ab_sents = " ".join(summary)

            # TODO: Get the most revelent sentences (highest rouge) from the src_text based on abstract
            summary_lists = separate_sentences(summary[0])
            text_lists = [[i] for i in sentence]
            summary_ids = greedy_selection(text_lists, summary_lists, len(summary_lists))
            src_summary = " ".join([" ".join(text_lists[i]) for i in summary_ids])

            try:
                tgt_img = figs[gold_figs[0].index(1)]
            except:
                tgt_img = figs[0]

            # example of gold_figs is [0, 0, 1, 0]
            record = {'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                      'duration': duration, 'figs': figs, 'gold_figs': gold_figs,
                      'tgt_img': tgt_img, 'src': sents, 'tgt': ab_sents, 'src_summary': src_summary}
            results[idx] = record
            idx += 1
    return results

class VATDataset(BaseDataset):
    def __init__(self, data_path, split, args_config, encoder=False, task=None):
        if data_path.endswith('.pkl'):
            self.data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.txt'):
            self.data = pd.read_table(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, mode='r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.data = process_video_meta(metadata)
        else:
            raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.split = split
        self.args_config = args_config
        self.encoder = encoder
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",
                                                             cache_dir='../preTrain_model/allenai/scibert_scivocab_uncased/')

        self.task_list = task

        super(VATDataset, self).__init__()

    def _load_data(self):
        self.text = []
        self.summarys = []
        self.image = []
        self.image_all = []
        self.audio = []
        self.video = []
        self.inputs_modal = []
        self.outputs_modal = []

        for task in self.task_list:
            inputs_modal = task['inputs_modal']
            self.inputs_modal = self.inputs_modal+inputs_modal
            outputs_modal = task['outputs_modal']
            self.outputs_modal = self.outputs_modal+outputs_modal
 
        self.inputs_modal = list(set(self.inputs_modal))  
        self.outputs_modal = list(set(self.outputs_modal)) 
        self.modality = self.inputs_modal + self.outputs_modal
        
        for _, row in self.data.items(): # get data from files

            self.summarys.append(row["tgt"])
            if "text" in self.modality:
                self.text.append(row["src_summary"])
            else:
                self.text.append(row["src_summary"])

            if "image" in self.modality:
                img_file2d = f'{self.args_config.data_path}/{self.args_config.data_name}/images_feat/' + row["vid"] + '.pkl'
                ground_truth_image = row["tgt_img"]

                with open(img_file2d, 'rb') as f:
                    image_dict = pickle.load(f)

                imgs = [v for k, v in image_dict.items()]
                img_slice = torch.cat(imgs, 0)

                idx = 0
                for k, v in image_dict.items():
                    if k == ground_truth_image:
                        label_idx = idx
                        gold_img = v
                    else:
                        idx += 1
                zeros = torch.zeros(1, img_slice.shape[1])
                if img_slice.shape[0] < 10:
                    for _ in range(10 - img_slice.shape[0]):
                        img_slice = torch.cat((img_slice, zeros), 0)
                elif label_idx < 10:
                    img_slice = img_slice[:10]
                elif label_idx >= 10:
                    img_slice = img_slice[:10 - 1]
                    img_slice = torch.cat((img_slice, gold_img), 0)
                self.image.append(gold_img.detach())
                self.image_all.append(img_slice.detach())

            if "audio" in self.modality:
                audio_dir = f'{self.args_config.data_path}/{self.args_config.data_name}/audio/'
                audio_slice = torch.from_numpy(np.load(os.path.join(audio_dir, row["vid"] + '.npy'), encoding='bytes'))
                if 43 < audio_slice.shape[0]:
                    audio_slice = audio_slice[:43]
                self.audio.append(audio_slice)

            if "video" in self.modality:
                video_dir = f'{self.args_config.data_path}/{self.args_config.data_name}/video/'
                self.video.append(torch.from_numpy(np.load(os.path.join(video_dir, row["vid"] + '.npy'), encoding='bytes')))

    
    def __getitem__(self, i):
        inputs_1 = {}

        if "image" in self.inputs_modal:
            inputs_1["image"] = self.image[i]

        if 'summary' in self.outputs_modal:
            inputs_1['Summary'] = self.tokenizer(
                self.summarys[i],
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

        if 'text' in self.inputs_modal:
            text = self.tokenizer(
                self.text[i],
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs_1['Text'] = text

        if 'audio' in self.inputs_modal:
            inputs_1['audio'] = self.audio[i]

        if 'video' in self.inputs_modal:
            inputs_1['video'] = self.video[i]

        inputs_1['text'] = self.text[i]
        inputs_1['summary'] = self.summarys[i]
        
        return inputs_1