# coding=utf-8
import cPickle
import numpy as np
import random
# import attention_textCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import cPickle
import os
import data2cv
import pcnn
import torch.nn.functional



def randvec(size, UNK=None):
    vec = []
    for i in range(size):
        vec.append(random.uniform(-1, 1))
    # print vec
    if UNK!=None:
        return UNK
    else:
        return vec


def gen_zerovec(size):
    vec = []
    for i in range(size):
        vec.append(0)
    return vec

def gen_UNK(size):
    vec = []
    for i in range(size):
        vec.append(random.uniform(-1, 1))
    # print vec
    cPickle.dump(vec, open('./UNKvec50.pkl', 'wb'))
    return vec



def string2arr(input, max_sentence_word, wordvector, wordvector_dim, UNK=None):
    batch_arr = []
    zerovec = gen_zerovec(wordvector_dim)
    for sentence in input:
        sentence_arr = []
        words = sentence.split()
        num_word = len(words)
        for i in range(min(num_word, max_sentence_word)):
            if wordvector.has_key(words[i]):
                sentence_arr.append(wordvector[words[i]])
            else:
                temvec = randvec(wordvector_dim, UNK)
                wordvector[words[i]] = temvec
                sentence_arr.append(temvec)
        if num_word < max_sentence_word:
            for i in range(num_word, max_sentence_word):
                sentence_arr.append(zerovec)
        batch_arr.append([sentence_arr])
    return batch_arr


if __name__ == '__main__':
    wordvector_data_path = '/home/hlg/data/glove/glove.6B.50d.txt'

    # save the model parameter
    timenow = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))

    runpath = '/home/yyj/pytorchCNN/PCNN/model/' + timenow + '/'
    mdir = os.mkdir(runpath)

    parameterlist = {}

    modelpath = runpath + timenow + '.pkl'
    parameter_path = runpath + timenow + '.para'


    traindatapath = '/home/yyj/pytorchCNN/PCNN/data'

    print 'model save in: ', modelpath
    print 'parameter save in: ', parameter_path

    parameterlist['localwordvectorpath'] = '/home/yyj/data/wordvector50_UNK.cpk'
    parameterlist['if_shuffle'] = True
    parameterlist['num_epochs'] = 200
    parameterlist['batch_size'] = 160
    parameterlist['max_sentence_word'] = 80
    parameterlist['wordvector_dim'] = 50
    parameterlist['PF_dim'] = 5
    parameterlist['relation_embedding_dim'] = 56  # one_hot 手动设置
    parameterlist['filter_size'] = 3
    parameterlist['num_filter'] = 230
    parameterlist['savewordvector'] = False
    parameterlist['loadwordvector'] = True
    parameterlist['userandomwordvector'] = False
    parameterlist['saverelationembedding'] = True
    parameterlist['loadlocalrelationembedding'] = True
    parameterlist['trainepoch'] = 25

    parameterlist['classes'] = 52

    parameterlist['trainbatchnow'] = 0
    parameterlist['trainepochnow'] = 0
    parameterlist['NArate'] = 1.0
    parameterlist['oversamping'] = 1

    # cPickle.dump(parameterlist, open(parameter_path, 'wb'))

    torch.cuda.set_device(0)

    print 'loading dataset.. ',
    if not os.path.isfile(traindatapath+'/test.p'):
        import dataset
        dataset.data2pickle(traindatapath+'/test_filtered.data', traindatapath+'/test.p')
    if not os.path.isfile(traindatapath+'/train.p'):
        import dataset
        dataset.data2pickle(traindatapath+'/train_filtered.data', traindatapath+'/train.p')

    testData = cPickle.load(open(traindatapath+'/test.p'))
    trainData = cPickle.load(open(traindatapath+'/train.p'))
    # testData = testData[1:5]
    # trainData = trainData[1:15]
    tmp = traindatapath.split('_')

    test = data2cv.make_idx_data_cv(testData, parameterlist['filter_size'], int(parameterlist['max_sentence_word']))
    train = data2cv.make_idx_data_cv(trainData, parameterlist['filter_size'], int(parameterlist['max_sentence_word']))
    print 'finished. '

    print 'load Wv ...  ',
    Wv = cPickle.load(open('/home/yyj/pytorchCNN/PCNN/data/Wv.p'))
    print 'finished.'

    rng = np.random.RandomState(3435)
    PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))


    # Wv = torch.FloatTensor(Wv)
    # print Wv[0]


    # print train[0].entities
    # print train[0].rel
    # print train[0].num
    # print train[0].sentences
    # print train[0].positions
    # print train[0].entitiesPos

    net = pcnn.textPCNN(parameterlist['max_sentence_word'], parameterlist['classes'], parameterlist['wordvector_dim'],
                        parameterlist['PF_dim'], parameterlist['filter_size'], parameterlist['num_filter'],
                        Wv, PF1, PF2).cuda()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

    np.random.seed(1234)
    epoch_now = 0
    batch_now = 0

    data_NA = []
    data_nonNA = []
    for bag in train:
        if bag.rel[0] == 0:
            data_NA.append(bag)
        else:
            data_nonNA.append(bag)



    print 'a epoch = %d batch' % (int(len(train))/int(parameterlist['batch_size']) + 1)
    for epoch in range(parameterlist['trainepoch']):
        print 'epoch = %d , start.. ' % epoch_now



        # shuffled_data = []
        # shuffle_indices = np.random.permutation(np.arange(len(train)))
        # for i in range(len(train)):
        #     shuffled_data.append(train[shuffle_indices[i]])


        shuffled_data_NA = []
        shuffled_data_nonNA = []

        shuffle_indices = np.random.permutation(np.arange(len(data_NA)))
        for i in range(len(data_NA)):
            shuffled_data_NA.append(data_NA[shuffle_indices[i]])

        shuffle_indices = np.random.permutation(np.arange(len(data_nonNA)))
        for i in range(len(data_nonNA)):
            shuffled_data_nonNA.append(data_nonNA[shuffle_indices[i]])


        # class_weight = []
        # for ii in range(52):
        #     if ii == 0:
        #         class_weight.append(1)
        #     else:
        #         class_weight.append(10)
        # class_weight = torch.FloatTensor(class_weight).cuda()

        num_NA = parameterlist['batch_size'] * 0.7
        num_nonNA = parameterlist['batch_size'] * 0.3

        NA_start = 0
        nonNA_start = 0

        no_next = False
        while True:
            # next_batch_start = bag_now + parameterlist['batch_size']
            # if next_batch_start<len(train):
            #     batch = shuffled_data[bag_now:next_batch_start]
            #     bag_now = next_batch_start
            # else:
            #     batch = shuffled_data[bag_now:len(train)]
            #     no_next = True
            if nonNA_start + num_nonNA < len(shuffled_data_nonNA):
                batch = shuffled_data_nonNA[int(nonNA_start):int(nonNA_start) + int(num_nonNA)] + shuffled_data_NA[int(NA_start):int(NA_start) + int(num_NA)]
                nonNA_start += num_nonNA
                NA_start += num_NA
            else:
                batch = shuffled_data_nonNA[int(nonNA_start):len(shuffled_data_nonNA)] + shuffled_data_NA[int(NA_start):int(NA_start) + int(num_NA)]
                no_next = True


            labels = []
            for bag in batch:
                labels.append(bag.rel[0])

            labels_arr = labels
            labels = Variable(torch.LongTensor(labels).cuda())


            # print '1'
            out = net(batch, parameterlist['max_sentence_word'], parameterlist['wordvector_dim'],
                      parameterlist['PF_dim'], parameterlist['num_filter'])
            # print '2'

            # print out
            loss = torch.nn.functional.cross_entropy(out, labels, weight=None)

            # loss = criterion(out, labels, weight=class_weight)
            optimizer.zero_grad()
            loss.backward()
            # print '3'
            optimizer.step()
            # print '4'

            batch_now += 1

            if batch_now % 1 == 0:
                print 'train batch = %d, last batch loss = %.3f' % (batch_now, loss.data[0])
                _, predicted = torch.max(out.data, 1)
                # print predicted.t()
                for j in range(len(labels_arr)):
                    print labels_arr[j], predicted[j][0], '    ',
                print '\n'

            if batch_now % 100 == 0:
                # if num_train_batch % epoch_batch == 0:
                print 'train epoch = %d, last batch loss = %.3f' % (epoch_now, loss.data[0])
                # if num_train_epoch % 10 == 0:
                torch.save(net.state_dict(), modelpath + str(batch_now) + '.pkl')
                cPickle.dump(parameterlist, open(parameter_path, 'wb'))



            if no_next:
                break

        # print 'epoch = %d , finished.. ' % epoch_now
        epoch_now += 1




    # batch = []
    # for i in range(8):
    #     batch.append(train[i])

    # out = net(batch, parameterlist['max_sentence_word'], parameterlist['wordvector_dim'], parameterlist['PF_dim'], parameterlist['num_filter'])







