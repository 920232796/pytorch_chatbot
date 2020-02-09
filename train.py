import math
import sys
import pickle
import time

from model import Hypothesis, ChatBot
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import handle_raw_data, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils
from config import data_file, batch_size, clip_grad, output_dir, vocab_file,embed_size, hidden_size, dropout_rate, lr, epoches

def train(train_data_src, train_data_tgt):
    """ Train the ChatBot Model.
    """
    train_data = list(zip(train_data_src, train_data_tgt))

    train_batch_size = int(batch_size)
    vocab = Vocab.load(vocab_file)
    model = ChatBot(embed_size=int(embed_size),
                hidden_size=int(hidden_size),
                dropout_rate=float(dropout_rate),
                vocab=vocab)
    model.train()

    uniform_init = float(0.1)
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.src))
    vocab_mask[vocab.src['<pad>']] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_trial = 0
    train_iter = patience = report_loss = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    begin_time = time.time()
    print('begin Maximum Likelihood training')
    log_every = 100
    while True:
        epoch += 1
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size_t = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size_t

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            
            report_examples += batch_size_t

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.5f ' \
                      'time elapsed %.2f sec' % (epoch, train_iter,
                                                report_loss / report_examples,
                                                time.time() - begin_time), file=sys.stderr)

                report_loss = report_examples = 0.

        ## 每一个epoch 保存一次模型
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_filename = save_path / f"model_{epoch:02}.pkl"
        torch.save(model.state_dict(), model_filename)

        if epoch == int(epoches):
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)

if __name__ == "__main__":
    train_data_src, train_data_tgt = handle_raw_data(data_file)
    print(len(train_data_src))
    print(len(train_data_tgt))
    save_src = []## 数据清洗
    save_tgt = []
    for index, each_data in enumerate(train_data_src):
        if (len(each_data)) < 3 or len(train_data_tgt[index]) < 1:
            continue 
        src_flag = 0
        tgt_flag = 0
        for i, word in enumerate(each_data):
            if word == "-" or word == "=" or word == "_" or word =="&" or word == "#" or word == "~":
            src_flag = 1
            break
        for word in train_data_tgt[index]:
            if word == "-" or word == "=" or word == "_" or word =="&" or word == "#" or word == "~":
            tgt_flag = 1
            break
        if src_flag == 1 or tgt_flag == 1:
            continue 

        save_src.append(each_data)
        save_tgt.append(train_data_tgt[index])
        
    print(len(save_src))
    print(len(save_tgt))

    train_data_src = save_src
    train_data_tgt = save_tgt
    ## 开始训练
    train(train_data_src, train_data_tgt)