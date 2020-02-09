import math
import sys
import pickle
import time

from model import ChatBot, ModelEmbeddings
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import handle_raw_data, batch_iter
from vocab import Vocab, VocabEntry
from model import Hypothesis
import torch
import torch.nn.utils

def decode():
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """
    model_path = "./output/model_13.pkl"
    test_data = [["你别难过"], ["你是谁呀"], ["你叫什么名字"], ["今天开不开心"], ["你喜欢谁"], \
        ["你是机器人把"], ["我觉得你有意思"], ["哈哈哈"], ["你知道我么"]]
    test_data_src = []
    for seq in test_data:
        data = [word for word in seq[0]]
        test_data_src.append(data)

    print(f"load model from {model_path}")
    model = ChatBot.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(5),
                             max_decoding_time_step=int(10) )

    # print(hypotheses)
    for hyps in hypotheses:
        print(hyps[0])
    # for src_sent, hyps in zip(test_data_src, hypotheses):
    #     top_hyp = hyps[0]
    #     hyp_sent = ' '.join(top_hyp.value)
    #     print(src_sent)
    #     print(hyp_sent)
    #     print("~~~~~~~~~~~")
        

def beam_search(model: ChatBot, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval() ## 因为使用了dropout层，通过这行代码就告诉模型现在是测试阶段，因此不会再去进行dropout

    hypotheses = []
    with torch.no_grad(): ## 这样就不用保存梯度了，提升速度 少用内存
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """

    decode()
  
if __name__ == '__main__':
    main()
