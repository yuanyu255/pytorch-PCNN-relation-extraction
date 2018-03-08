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



if __name__ == '__main__':
    #torch.cuda.set_device(3)

    # save the model parameter
    timenow = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    runpath = './model/' + timenow + '/'
    mdir = os.mkdir(runpath)

    parameterlist = {}

    modelpath = runpath + timenow + '.pkl'
    parameter_path = runpath + timenow + '.para'


    traindatapath = './data'

    print 'model save in: ', modelpath
    print 'parameter save in: ', parameter_path


    parameterlist['if_shuffle'] = True
    parameterlist['trainepoch'] = 50
    parameterlist['batch_size'] = 160
    parameterlist['max_sentence_word'] = 80
    parameterlist['wordvector_dim'] = 50
    parameterlist['PF_dim'] = 5
    parameterlist['filter_size'] = 3
    parameterlist['num_filter'] = 230
    parameterlist['classes'] = 52

    # cPickle.dump(parameterlist, open(parameter_path, 'wb'))



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
    Wv = cPickle.load(open('./data/Wv.p'))
    print 'finished.'

    rng = np.random.RandomState(3435)
    PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))


    net = pcnn.textPCNN(parameterlist['max_sentence_word'], parameterlist['classes'], parameterlist['wordvector_dim'],
                        parameterlist['PF_dim'], parameterlist['filter_size'], parameterlist['num_filter'],
                        Wv, PF1, PF2)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

    np.random.seed(1234)
    epoch_now = 0
    batch_now = 0
    print 'a epoch = %d batch' % (int(len(train))/int(parameterlist['batch_size']) + 1)
    for epoch in range(parameterlist['trainepoch']):
        print 'epoch = %d , start.. ' % epoch_now
        shuffled_data = []
        shuffle_indices = np.random.permutation(np.arange(len(train)))
        for i in range(len(train)):
            shuffled_data.append(train[shuffle_indices[i]])
        bag_now = 0

        class_weight = []
        for ii in range(52):
            if ii == 0:
                class_weight.append(1)
            else:
                class_weight.append(10)
        class_weight = torch.FloatTensor(class_weight)

        no_next = False
        while True:
            next_batch_start = bag_now + parameterlist['batch_size']
            if next_batch_start<len(train):
                batch = shuffled_data[bag_now:next_batch_start]
                bag_now = next_batch_start
            else:
                batch = shuffled_data[bag_now:len(train)]
                no_next = True

            labels = []
            for bag in batch:
                labels.append(bag.rel[0])

            labels_arr = labels
            labels = Variable(torch.LongTensor(labels))


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

            if batch_now % 10 == 0:
                print 'train batch = %d, last batch loss = %.3f' % (batch_now, loss.data[0])
                _, predicted = torch.max(out.data, 1)
                # print predicted.t()
                for j in range(len(labels_arr)):
                    print labels_arr[j], predicted[j], '    ',
                print '\n'

            if batch_now % 500 == 0 or batch_now == 1:
                # if num_train_batch % epoch_batch == 0:
                print 'train epoch = %d, last batch loss = %.3f' % (epoch_now, loss.data[0])
                # if num_train_epoch % 10 == 0:
                torch.save(net, modelpath + str(batch_now) + '.pkl')
                cPickle.dump(parameterlist, open(parameter_path, 'wb'))

                # print 'load..'
                # #net.load_state_dict(torch.load(net.state_dict(), modelpath + str(batch_now) + '.pkl'))
                # print 'finish'



            if no_next:
                break

        # print 'epoch = %d , finished.. ' % epoch_now
        epoch_now += 1





