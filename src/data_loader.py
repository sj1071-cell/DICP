import torch
import numpy as np
import random
import pickle
import os
import copy
import json

from tqdm import tqdm
from transformers import RobertaModel, BertModel

from utils.logger import get_logger

logger = get_logger()


def load_dict(file_path):
    with open(file_path, 'r') as f:
        dict_ = {line.rstrip('\n'): idx for idx, line in enumerate(f.readlines())}

    return dict_


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        pkl_data = pickle.load(f)

    return pkl_data

def get_random_sentences(filename, num=2):
    """从JSON文件中随机选择指定数量的不重复句子"""
    with open(filename, 'r', encoding='utf-8') as file:
        sentences = json.load(file)
    return random.sample(sentences, min(num, len(sentences)))


class Dataset(object):
    def __init__(self, args, dataset, tokenizer):
        super(Dataset, self).__init__()

        self.batch_size = args.batch_size
        self.y_label = {
            'NULL': 0,
            'null': 1,  # plot link, but no direction
            'plot_null': 1,
            'FALLING_ACTION': 1,
            'PRECONDITION': 1,
            'Coref': 1,
            'Cause-Effect': 1,
            'Cause-Effect1': 1,
            'Cause-Effect2': 1,
            'Cause': 1
        }  # data label

        self.args = args
        self.tokenizer = tokenizer  # bert tokenizer
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))
        self.mask_token = tokenizer.mask_token
        self.tokenize_data = self.tokenize(dataset)
        self.dataset = dataset
        self.demonencoder = self.demonencoder(dataset)

    def demonencoder(self, batch_samples):
        num = 0
        # 记录每一层的前缀示例
        prefix_demon = {}
        random_sentences = get_random_sentences('demons.json', self.args.demon_num)
        for item in batch_samples:
            prefix_demon[num] = random_sentences
            num += 1
        return prefix_demon

    def reader(self, device, shuffle=False):
        """
        read dataset
        :param device: model used device
        :param shuffle: every epoch shuffle
        :return: None
        """
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)

            batch_samples = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            batch_tokenize = [self.tokenize_data[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            batch_demos = [self.demonencoder[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch_samples, batch_tokenize, batch_demos, device)
            # return [self.batchify(batch_samples, batch_tokenize, batch_graphs, batch_align, batch_path, device)]
        if shuffle:
            random.shuffle(self.shuffle_list)
            logger.info("Data shuffle finish.")

    def tokenize(self, batch_samples):
        """
        convert tokens to id
        :param batch_samples: tokens batch data
        :return: tokenized batch data
        """
        batch_tokenize = []
        for item in batch_samples:
            sent_s = copy.deepcopy(item[1])
            span1 = item[2]
            span2 = item[3]

            e1_start = span1[0]
            e1_end = span1[-1] + 1
            e2_start = span2[0]
            e2_end = span2[-1] + 1

            event1 = sent_s[e1_start:e1_end]
            event2 = sent_s[e2_start:e2_end]
            event1_words = ' '.join(event1)
            event2_words = ' '.join(event2)

            offset = 2 if e1_start < e2_start else 0
            sent_s.insert(e1_start, '<e1>')
            sent_s.insert(e1_end + 1, '</e1>')
            sent_s.insert(e2_start + offset, '<e2>')
            sent_s.insert(e2_end + 1 + offset, '</e2>')

            if (e1_start < e2_start):
                prompts = sent_s + [event1_words] + \
                          ["<t1>", self.mask_token, "</t1>"] + \
                          [event2_words + "."]
            else:
                prompts = sent_s + [event2_words] + \
                          ["<t1>", self.mask_token, "</t1>"] + \
                          [event1_words + "."]
            # prompts = sent_s + [event1_words] +\
            #           ["<t1>", self.mask_token, "</t1>"] + \
            #         [event2_words + "."]
            # ["A man suspected of shooting three people ,killing one , at an accounting firm where was fired last week was arrested after a high - speed chase a few hours after the Monday morning attack , authorities said.killing </causal> arrested."] +\
            # ['As many SA gamers may have  noticed, SEACOM is experiencing downtime again. noticed </na> experiencing.']
            sent_s = ' '.join(prompts)

            batch_tokenize.append([item[0], sent_s, item[4]])

        return batch_tokenize


    def batchify(self, batch_samples, batch_tokenize, batch_demos, device):
        """
        padding batch data
        :param batch: tokenized batch data
        :param device: model used device
        :return: batch data tensor
        """

        sentences_s, data_y = [], []
        batch_idx = []
        batch_mask = []
        mask_indices = []

        for data in batch_tokenize:
            sentence = data[1]
            y = self.y_label[data[2]] if data[2] in self.y_label else 0
            data_y.append(y)
            encode_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                padding='max_length',
                max_length=100,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            arg_1_idx = encode_dict['input_ids']
            arg_1_mask = encode_dict['attention_mask']

            if len(batch_idx) == 0:
                batch_idx = arg_1_idx
                batch_mask = arg_1_mask
                # mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
                mask_indices = torch.nonzero(arg_1_idx == 103, as_tuple=False)[0][1]
                mask_indices = torch.unsqueeze(mask_indices, 0)
            else:
                batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
                batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
                mask_indices = torch.cat(
                    # (mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
                    (mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 103, as_tuple=False)[0][1], 0)), dim=0)

        return batch_idx, batch_mask, mask_indices, torch.LongTensor(data_y).to(device), batch_demos