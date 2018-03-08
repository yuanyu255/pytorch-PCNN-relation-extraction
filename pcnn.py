# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable


class textPCNN(torch.nn.Module):
    def __init__(self, sequence_length, num_sentences_classes,
                 word_embedding_dim, PF_embedding_dim,
                 filter_size, num_filters,
                 word_embedding, PF1_embedding, PF2_embedding):
        super(textPCNN, self).__init__()

        self.conv = nn.Conv2d(1, num_filters, (filter_size, word_embedding_dim + 2*PF_embedding_dim))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(num_filters * 3, num_sentences_classes)

        self.embedding_wv = nn.Embedding(sequence_length, word_embedding_dim)
        self.embedding_PF1 = nn.Embedding(PF_embedding_dim, word_embedding_dim)
        self.embedding_PF2 = nn.Embedding(PF_embedding_dim,word_embedding_dim)

        self.wordvec = Parameter(torch.FloatTensor(word_embedding))
        self.PF1_embedding = Parameter(torch.FloatTensor(PF1_embedding))
        self.PF2_embedding = Parameter(torch.FloatTensor(PF2_embedding))



    def forward(self, input_, sentence_word, word_embedding_dim, PF_embedding_dim, num_filters, if_eval=False):
        batch_size = len(input_)
        sentence_num = 0
        for bag in input_:
            sentence_num += bag.num


        self.embedding_wv.weight = self.wordvec
        self.embedding_PF1.weight = self.PF1_embedding
        self.embedding_PF2.weight =self.PF2_embedding

        sentence_all = []
        sentence_PF1_all = []
        sentence_PF2_all = []

        entitypos1 = []
        entitypos2 = []
        num_sentence = 0
        for bag in input_:
            sentence_all += [sentence for sentence in bag.sentences]
            sentence_PF1_all += [sentence_PF[0] for sentence_PF in bag.positions]
            sentence_PF2_all += [sentence_PF[1] for sentence_PF in bag.positions]
            for i in range(bag.num):
                entitypos1_ = bag.entitiesPos[i][0]
                entitypos2_ = bag.entitiesPos[i][1]
                entitypos1.append(entitypos1_)
                entitypos2.append(entitypos2_)
                num_sentence += 1


        sentence_embedding = self.embedding_wv(Variable(torch.LongTensor(sentence_all)))
        sentence_PF1_enbedding = self.embedding_PF1(Variable(torch.LongTensor(sentence_PF1_all)))
        sentence_PF2_embedding = self.embedding_PF2(Variable(torch.LongTensor(sentence_PF2_all)))

        batch_input = torch.cat((sentence_embedding, sentence_PF1_enbedding, sentence_PF2_embedding), 2)
        batch_input = torch.unsqueeze(batch_input, 1)


        conv = self.conv(batch_input)
        # print conv.size()

        conv = self.tanh(conv)

        for i in range(num_sentence):
            pool1 = torch.nn.functional.max_pool2d(torch.unsqueeze(conv[i], 0)[:, :, :entitypos1[i] + 1], (entitypos1[i]+1, 1))
            pool2 = torch.nn.functional.max_pool2d(torch.unsqueeze(conv[i], 0)[:, :, entitypos1[i]:entitypos2[i] + 1], (entitypos2[i] - entitypos1[i] + 1, 1))
            pool3 = torch.nn.functional.max_pool2d(torch.unsqueeze(conv[i], 0)[:, :, entitypos2[i]:], (sentence_word - entitypos2[i], 1))

            pool1 = torch.squeeze(pool1, 2)
            pool1 = torch.squeeze(pool1, 2)
            pool2 = torch.squeeze(pool2, 2)
            pool2 = torch.squeeze(pool2, 2)
            pool3 = torch.squeeze(pool3, 2)
            pool3 = torch.squeeze(pool3, 2)

            pool_all = torch.cat((pool1, pool2, pool3), 0)
            # print pool_all.size()

            sentence_feature = torch.t(pool_all).clone().resize(1, 3*num_filters)
            # print sentence_feature.size()
            if i == 0:
                bag_sentence_feature = sentence_feature
            else:
                bag_sentence_feature = torch.cat((bag_sentence_feature ,sentence_feature), 0)



        scores = self.output(bag_sentence_feature)
        scores_out = self.output(self.dropout(bag_sentence_feature))


        if if_eval:
            return scores

        # print scores.size()

        sentence_begin = 0
        bag_now = 0
        select = []
        for bag in input_:
            sentence_end = sentence_begin + bag.num
            rel = bag.rel[0]
            max_index = -1
            max_score = -999999
            for j in range(sentence_begin, sentence_end):
                if scores[j][rel].data[0] > max_score:
                    max_score = scores[j][rel].data[0]
                    max_index = j
            select.append(max_index)

            bag_now += 1
            sentence_begin = sentence_end
        select = Variable(torch.LongTensor(select))
        # print select
        output = torch.index_select(scores_out, 0, select)


        return output





















