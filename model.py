#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from embedding import ModelEmbeddings

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class ChatBot(nn.Module):
    """ Simple Chat Bot Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init ChatBot Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(ChatBot, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.encoder = None
        self.decoder = None
        self.h_projection = None## 将encoder双向的hidden特征，线性计算为单向特征～
        self.c_projection = None## 同h_projection 一样
        self.att_projection = None ## 将encoder双向输出 线性计算为单项特征
        self.combined_output_projection = None ## 当前decoder输出连接attention层输出的时候需要进行线性计算
        self.target_vocab_projection = None
        self.dropout = None

        self.encoder = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, bias=True)
        self.decoder = nn.LSTMCell(embed_size + self.hidden_size, self.hidden_size, bias=True)
        ### h_projection: 这个是初始化decoder初始 hidden 状态用的，因为编码器最终hidden是双向的，因此 2 * hidden_size
        ### c_projection: 这个也差不多，是初始化decoder初始的 cell 状态用的。
        self.h_projection = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.att_projection = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.src), bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.src.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        ## 给encoder 产生 mask encoder 输入的一句话，pad的部分为1，其余部分为0.
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        ## combined_outputs: (time_step, batch_size, hidden_size) 
        ## 这个是decoder当前时间步输出跟attention联合起来计算得到的输出～
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        ## 这里就是计算每个词概率，如果softmax之后，很小，那么log之后就是负无穷，说明肯定不是这个词
        ## (time_step, batch_size, vocab_size)
        ## 注意P是预测的结果得分，他是从<s>开始预测的！因此结果里面肯定不包含第一个时间步！！！
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        ## 为了把pad的地方直接改成0
        target_masks = (target_padded != self.vocab.src['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ## 首先把已经pad之后的tensor 输入到embedding层，进行 embedding 输出:(time_step, batch_size, embed_size)
        X = self.model_embeddings.source(source_padded)
        ## 输入到LSTM里面的时候，进行一下这个操作，因此LSTM算到pad的地方就停止了，不往下计算了～
        X = pack_padded_sequence(X, source_lengths)
        ## 输入到LSTM里面
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        ##  再进行还原一下～也就是pack解开 这个以后需要自己再实验一下～
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, padding_value=self.vocab.src["<pad>"])
        ## 转置一下输出结果，转置之后为batch_size在前了！ (batch_size, time_step, 2 * hidden_size)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)
        ## 注意LSTM中，last_hidden 输出的时候维度为：(num_layers * num_directions, batch_size, hidden_size)
        ## 所以这里是把last_hidden的hidden size 维度连接起来 成为 2 * hidden_size
        ## 因此 h_projection 是 Linear(2 * hidden_size, hidden_size) 
        ## 跟以前不同的是 这里是把双向特征 联合一个 参数矩阵 运算了一下 才作为decoder的初始状态
        ## 以前是直接把encoder 最终单向特征作为decoder初始状态
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[1], last_hidden[0]), 1))
        ## 这里c也是同理～
        init_decoder_cell = self.c_projection(torch.cat((last_cell[1], last_cell[0]), 1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        ### END YOUR CODE

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        ## 这里为啥要切掉最后元素？ 因为训练的时候，decoder 到 倒数第二个元素，最后一个元素就没必要接着输入了，因为已经到最后了～
        target_padded = target_padded[:-1]

        ## 初始化decoder状态
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        ## enc_hiddens就是编码器encoder的输出，并且经过转置的 维度为：(batch_size, time_step, hidden_size)
        batch_size = enc_hiddens.size(0)
        ## 初始化上一个时间步的最终输出
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ## 把enc_hidden(batch_size, time_step, 2 * hidden_size) 输入到atten层
        ## 输出 (batch_size, time_step, hidden_size)
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        ## 输入到embedding 层
        Y = self.model_embeddings.target(target_padded)
        ## Y是所有时间步的，(time_step, batch_size, embed_size)
        ## 因此需要split 分割，每次取一个时间步，因此第二个参数为1
        for Y_t in torch.split(Y, 1, dim=0):
            Y_t= Y_t.squeeze(0)
            ## 这里我们不使用强制教学了，而是将上一个时间步的输出跟当前真正输入在特征层连接起来，这样就告诉了decoder上次可能生成错误了
            ## y_bat_t:(batch_size, hidden_size + embed_size)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            ## 得到真正的输出o_t 和 decoder state
            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        ## 默认dim=0，因此结果其实是(time_step, batch_size, hidden_size)
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        dec_state = self.decoder(Ybar_t, dec_state)
        ## dec_hidden 是decoder当前输出 (batch_size, hidden_size)
        dec_hidden, dec_cell = dec_state
        ## enc_hidden_proj 是 encoder所有双向输出 变为单向(batch_Size, time_step, hidden_size)
        ## 通过当前decoder输出 跟 所有encoder输出 运算，得出来encoder每个时间步对当前decoder输出的影响程度
        ## e_t ：(batch_size, time_step)
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(-1)).squeeze(-1)
        # Set e_t to -inf where enc_masks has 1
        ## 把encoder pad的部分设置为 负无穷，这样进行softmax 的时候，就为0，也就是pad对当前deocder输出没有影响
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=1)
        ##  a_t = (batch_size, 1, time_step).dot( (batch_size, time_step, 2*hidden_Size) )
        ## a_t :(batch_size, 2*hidden_size) 也就是attention的输出
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
        ## 连接attention特征层和当前decoder输出
        U_t = torch.cat((a_t, dec_hidden), dim=1)
        ## 经过一个线性层计算
        ## V_t: (batch_size, hidden_size)
        V_t = self.combined_output_projection(U_t)
        ## 加一下tanh非线性化， drop 正则化，得到真正的输出～
        O_t = self.dropout(torch.tanh(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        ## 将输入进来的数据转换成tensor 而且转置了 time_step在前面～
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        ## 通过编码器 进行特征提取
        ## src_encodings: (batch_size, time_step, 2 * hidden_size)
        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        ## 通过attention计算～
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.src['</s>']
        ## 从开始符号开始预测
        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        ## 循环每个时间步
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            ## 这行就是 当前时间步同时预测几个序列，刚开始从[["<s>"]]开始 就是只预测一个，第二个时间步则要预测k个
            hyp_num = len(hypotheses)
            ## 因为若同时预测多个，也就是当前decoder输入有多个，那么encoder输出也要有多个，并且是一样的
            ## 因此用expand就ok了～
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
            ## 这里也是 encoder做attention第一步计算的结果也要扩大
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            ## 得到n个预测的上一个时间步的输出，因此要取-1，作为n个当前decoder的输入
            y_tm1 = torch.tensor([self.vocab.src[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            ## 将当前时间步n个decoder输入进行词嵌入
            y_t_embed = self.model_embeddings.target(y_tm1)
            ## x就是去连接一下 当前时间步的输入Y_t:(batch_size, embed_size)和上一个时间步step函数的输出o_prev,也就是下面的att_t
            ## 这个o_prev(batch_size, hidden_size)就是通过上一个时间步的x跟attention计算得出来的
            ## x: (batch_size, embed_size + hidden_size)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            ## 其中 att_t 就是新计算出来的att_tm1:(b, h)
            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)
            # log probabilities over target words
            ## log_p_t: (b, v) n个decoder输入 所得到n个输出得分情况～其中每个又对应vocab—size个分值
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            ## 看看还需要取tok几。
            live_hyp_num = beam_size - len(completed_hypotheses)
            ## 叠加得分
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            ## 取topk个最大的，k取决于现在完成几个预测了 比如一共要五个，但是已经完成四个了，
            # 那么就直接每次时间步都找最大的那个概率就行了～不用找前两个概率最大的了～
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            ## 因为我们是同时预测n个输出，取topk的时候用view(-1)把n个情况展平了,比如矩阵为（4 * 1024）那么展平之后便为(4096),然后取topk
            ## 那么这里就要得到 hyp_id就是预测的n中第几个
            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.src)
            ## 这里就是每一个的具体是哪个单词，根据这两个变量我们就可以定位同时预测的n个序列，每个序列输出的是哪几个单词了～
            ## 因为是展平取topk，那么也可能有的序列取到有多个单词，有的序列没有取到单词
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.src)

            ## 定义新预测序列
            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            ## 因为上面基于n个序列同时预测下一个单词，也会预测n个新结果（如果没有结束的序列）
            ## 那么这里就是对n个新结果进行处理，并判断有没有</s>，也就是结束符
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                ## 通过这三个变量，便可以定位这个得分是第几个序列的第几个单词的得分了～
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()
                ## 得到预测的单词
                hyp_word = self.vocab.src.id2word[hyp_word_id]
                ## 把新预测的单词加到对应的序列里面去
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    ## 证明一个序列已经预测完成了～
                    ## 过滤到<s>和</s> 然后把结果加入到最终结果集里面去
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    ## 否则的话，还没结束，就继续把n个序列保存好，进行下一个时间步的循环～
                    new_hypotheses.append(new_hyp_sent)
                    ## 这个id意思是n个序列里面的第几个 定位用的，下面也会用到，因为不同的序列，隐藏状态也不同，这个id就是为了确定下一个时间步需要用哪几个隐藏状态的
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break
            ## 将id转为 tensor变量
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            ## 确定隐藏状态hidden 和 cell
            ## h_t: (b, h)
            ## h_tm1: (b, h) 只是可能有的序列重复取了，有的序列就没了，总数不变（如果没有得到</s>序列的话）
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        import config
        from vocab import Vocab, VocabEntry
        ## 加载词典
        vocab = Vocab.load("./vocab.json")
        # print(vocab.src)
        model = ChatBot(embed_size=config.embed_size, hidden_size=config.hidden_size, vocab=vocab)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
