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
from sklearn.metrics import precision_recall_curve
import re
import datetime
from operator import itemgetter, attrgetter



def f_map(a, b):
    return [a, b]

if __name__ == '__main__':

    modelpath = ''
    #### .pkl file ##########

    parameter_path = ''
    #### .para file #############



    parameterlist = cPickle.load(open(parameter_path, 'rb'))
    traindatapath = './data'
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
    Wv = cPickle.load(open('/home/slotFilling/PCNN_baseline_release/data/Wv.p'))
    print 'finished.'

    rng = np.random.RandomState(3435)
    PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]))
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))

    max_2 = 0
    print 'eval :', modelpath
    while not os.path.isfile(modelpath):
        print 'sleep .... zzz .. zz . z .'
        time.sleep(60)
        # print modelpath, ' is not exist'
        # continue
    time.sleep(20)

    print 'load the net ...',

    net = torch.load(modelpath)
    print '  finished.'
    test_label_true = []
    test_label_prob = []

    np.random.seed(1234)
    epoch_now = 0
    batch_now = 0
    tp = 0
    fp = 0
    fn = 0
    print 'a epoch = %d batch' % (int(len(test))/int(parameterlist['batch_size']) + 1)
    for epoch in range(1):
        print 'epoch = %d , start.. ' % epoch_now
        shuffled_data = []
        shuffle_indices = np.random.permutation(np.arange(len(test)))
        for i in range(len(test)):
            shuffled_data.append(test[shuffle_indices[i]])
        bag_now = 0


        no_next = False
        while True:
            # print 0
            next_batch_start = bag_now + parameterlist['batch_size']
            if next_batch_start<len(test):
                batch = shuffled_data[bag_now:next_batch_start]
                bag_now = next_batch_start
            else:
                batch = shuffled_data[bag_now:len(test)]
                no_next = True

            labels = []
            for bag in batch:
                labels.append(bag.rel[0])

            labels_arr = labels
            labels = Variable(torch.LongTensor(labels))

            out = net(batch, parameterlist['max_sentence_word'], parameterlist['wordvector_dim'],
                      parameterlist['PF_dim'], parameterlist['num_filter'],
                      if_eval=True)

            output_norm = torch.nn.functional.softmax(out).cpu()
            _, predicted = torch.max(out.data, 1)


            batch_now += 1

            # if batch_now%5 == 0:
            #     print 'batch = ', batch_now

            t_1 = datetime.datetime.now()
            sentence_begin = 0
            for i in range(len(batch)):
                sentence_end = sentence_begin + batch[i].num
                prob = []

                t_3 = datetime.datetime.now()

                for ii in range(52):
                    p = output_norm[sentence_begin:sentence_end, ii:ii + 1].max().data[0]
                    prob.append(p)
                # prob = output_norm[sentence_begin:sentence_end, batch_labels[i]:batch_labels[i]+1].max().data[0]
                # print prob
                # print prob

                t_4 = datetime.datetime.now()



                for j in range(52):
                    if j != 0:
                        # print j, batch[i].rel[0]
                        # if (j == batch[i].rel[0] or j == batch[i].rel[1] or j == batch[i].rel[2] or j == batch[i].rel[3]):
                        if j == batch[i].rel[0]:
                        # if j == batch[i].rel[label_]:
                            test_label_true.append(1)
                            test_label_prob.append(prob[j])
                        else:
                            test_label_true.append(0)
                            test_label_prob.append(prob[j])
                sentence_begin += batch[i].num

            t_2 = datetime.datetime.now()

            # print 't2-t1 = ', t_2 - t_1
            # print 't2-t4 = ', t_2 - t_4
            # print 't4-t3 = ', t_4 - t_3

            if batch_now % 100 == 0 or batch_now == 2:
                precision, recall, thresholds = precision_recall_curve(test_label_true, test_label_prob)

                outfile = open(modelpath + '.PR', 'w')
                for i in range(len(precision)):
                    if recall[i] <= 0.8:
                        if i < len(precision) - 1:
                            tem = '%-15s   %-15s  %-15f\n' % (recall[i], precision[i], thresholds[i])
                        else:
                            tem = '%-15s   %-15s\n' % (recall[i], precision[i])
                        outfile.write(tem)
                outfile.close()


            sentence_begin = 0
            for i in range(len(batch)):
                sentence_end = sentence_begin + batch[i].num
                pred_labels = []
                for j in range(sentence_begin, sentence_end):
                    pred_labels.append(predicted[j])


                if batch[i].rel[0] != 0:
                    if batch[i].rel[0] in pred_labels:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if 0 not in pred_labels:
                        fp += 1

                sentence_begin += batch[i].num


            if tp + fp > 0:
                prec = float(tp) / (tp + fp)
            else:
                prec = 0
            if tp + fn > 0:
                recall = float(tp) / (tp + fn)
            else:
                recall = 0
            temst = 'batch = %-7d  tp = %-7d  fp = %-7d  fn = %-7d  prec = %-10.4f  recall = %-10.4f  ' % (batch_now, tp, fp, fn, prec, recall)
            if batch_now % 50 == 0 or batch_now == 1:
                print temst

            # outfile = open(modelpath + '.pr', 'w')
            # outfile.write(temst)
            # outfile.close()




            if no_next:
                break

        # print 'epoch = %d , finished.. ' % epoch_now
        epoch_now += 1

        pred = map(f_map, test_label_true, test_label_prob)
        s_pred = sorted(pred, key=itemgetter(1), reverse=True)
        t = 0.
        f = 0.
        p100 = 0.
        p200 = 0.
        p300 = 0.
        for ttt in range(300):
            if s_pred[ttt][0] == 1:
                t += 1
            else:
                f += 1
            if ttt == 99:
                p100 = t / (t + f)
                print 'P@100 = ', t / (t + f)
            if ttt == 199:
                p200 = t / (t + f)
                print 'P@200 = ', t / (t + f)
            if ttt == 299:
                p300 = t / (t + f)
                print 'P@300 = ', t / (t + f)


        precision, recall, thresholds = precision_recall_curve(test_label_true, test_label_prob)
        os.remove(modelpath + '.PR')
        recall_0_1_tem = 0
        rec1 = 0
        recall_0_2_tem = 0
        rec2 = 0
        recall_0_3_tem = 0
        rec3 = 0
        for i in range(len(precision)):
            if recall_0_1_tem == 0 and recall[i]>0.1:
                rec1 = precision[i]
            else:
                recall_0_1_tem = 1
            if recall_0_2_tem == 0 and recall[i]>0.2:
                rec2 = precision[i]
            else:
                recall_0_2_tem = 1
            if recall_0_3_tem == 0 and recall[i]>0.3:
                rec3 = precision[i]
            else:
                recall_0_3_tem = 1

        name = '%s_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f.PR' % (modelpath, rec1, rec2, rec3, p100, p200, p300)
        outfile = open(name, 'w')
        for i in range(len(precision)):
            if recall[i] <= 0.6:
                if i < len(precision) - 1:
                    tem = '%-15s   %-15s  %-15f\n' % (recall[i], precision[i], thresholds[i])
                else:
                    tem = '%-15s   %-15s\n' % (recall[i], precision[i])
                outfile.write(tem)
        outfile.close()










