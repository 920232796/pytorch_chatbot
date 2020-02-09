#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path  

def handle_raw_data(path: str) -> List[Tuple[List[str]]]:
    """
    @param path: raw data path 
    @param build the data file from raw data
    @return data: [(['呵', '呵'], ['<s>', '是', '王', '若', '猫', '的', '。', '</s>']),]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        line_data = f.readlines()
    
    # print(line_data[:])
    # 数据是以单独的E\n 作为分割一组对话的！
    conversations = []
    input_seq = []
    target_seq = []
    flag = -1
    for index, line in enumerate(line_data):
        if line.strip() == "E":
          flag = 0
          continue
        if flag == 0:
            ## 证明是输入句子
            line = line.lstrip("M ").rstrip("\n")## 简单处理一下str
            temp = []
            for word in line:
                temp.append(word)
            input_seq.append(temp)
            flag = 1
            continue
        if flag == 1:
            ## 证明是目标句子 需要添加开始符和结束符
            line = line.lstrip("M ").rstrip("\n")## 简单处理一下str
            temp = []
            for word in line:
                temp.append(word)
            temp = ["<s>"] + temp + ["</s>"]
            target_seq.append(temp)
            flag = -1
    
    return input_seq, target_seq

## 给数据加pad，使得所有的数据一样长
def pad_sents(sents: List[List[str]], pad_token: str):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    max_length = max([len(sent) for sent in sents])
    sents_padded = [sent + [pad_token] * max(0, max_length - len(sent)) for sent in sents]

    return sents_padded

# ## 读语料，返回列表数据，如果是target语料，则需要添加<s> </s>
# def read_corpus(file_path: str, source: str) -> List[List[str]]:
#     """ Read file, where each sentence is dilineated by a `\n`.
#     @param file_path (str): path to file containing corpus
#     @param source (str): "tgt" or "src" indicating whether text
#         is of the source language or target language
#     """
#     data = []
#     for line in open(file_path):
#         sent = line.strip().split(' ')
#         # only append <s> and </s> to the target sentence
#         if source == 'tgt':
#             sent = ['<s>'] + sent + ['</s>']
#         data.append(sent)

#     return data


def batch_iter(data: List[Tuple[List[str]]], batch_size: int, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.floor(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        ## 输入到网络的时候，必须按照句子长度从大到小进行输入～
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


if __name__ == "__main__":
    data = handle_raw_data("data/data.conv")
    print(data)