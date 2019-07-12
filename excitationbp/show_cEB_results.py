### Print digit and goal prediction message and plot examples
### This code script was completely written by us, who are anonymous authors of the submitted paper under review.

import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys, copy ; sys.path.append('..')
from predict_w_cEB import obtain_example_stats

reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()

if torch.cuda.is_available():
    useCuda = True
else:
    useCuda = False

def print_each_brief_stats(avg_stats,pair_amount,stats_amount=0):
    [digit_correct, parity_correct, highlow_correct, target_digit_prob, target_parity_prob, target_highlow_prob, \
     pred_digit_prob, pred_parity_prob, pred_highlow_prob, \
     nontarget_max_digit_prob, nontarget_max_parity_prob, nontarget_max_highlow_prob, \
     target_digit, target_parity, target_highlow, \
     pred_digit, pred_parity, pred_highlow, \
     nontarget_max_digit, nontarget_max_parity, nontarget_max_highlow] = avg_stats
    digit_acc_msg = '"digit" acc: {:.3f}'.format(digit_correct)
    target_digit_prob_msg = 'lbl prob: {:.3f}'.format(target_digit_prob)
    pred_digit_prob_msg = 'pred prob: {:.3f}'.format(pred_digit_prob)
    if stats_amount == 0:
        even_acc_msg = '"even" acc: {:.3f}'.format(parity_correct)
        target_even_prob_msg = 'lbl prob: {:.3f}'.format(target_parity_prob)
        pred_even_prob_msg = 'pred prob: {:.3f}'.format(pred_parity_prob)
        print('[even] {}, {}, {} | {}, {}, {}'.format(digit_acc_msg,target_digit_prob_msg,pred_digit_prob_msg,\
                                                              even_acc_msg,target_even_prob_msg,pred_even_prob_msg))
    elif stats_amount == 1:
        odd_acc_msg = '"odd" acc: {:.3f}'.format(parity_correct)
        target_odd_prob_msg = 'lbl prob: {:.3f}'.format(target_parity_prob)
        pred_odd_prob_msg = 'pred prob: {:.3f}'.format(pred_parity_prob)
        print('[odd] {}, {}, {} | {}, {}, {}'.format(digit_acc_msg,target_digit_prob_msg,pred_digit_prob_msg,\
                                                              odd_acc_msg,target_odd_prob_msg,pred_odd_prob_msg))
    elif stats_amount == 2:
        low_acc_msg = '"low" acc: {:.3f}'.format(highlow_correct)
        target_low_prob_msg = 'lbl prob: {:.3f}'.format(target_highlow_prob)
        pred_low_prob_msg = 'pred prob: {:.3f}'.format(pred_highlow_prob)
        print('[low] {}, {}, {} | {}, {}, {}'.format(digit_acc_msg,target_digit_prob_msg,pred_digit_prob_msg,\
                                                              low_acc_msg,target_low_prob_msg,pred_low_prob_msg))
    elif stats_amount == 3:
        high_acc_msg = '"high" acc: {:.3f}'.format(highlow_correct)
        target_high_prob_msg = 'lbl prob: {:.3f}'.format(target_highlow_prob)
        pred_high_prob_msg = 'pred prob: {:.3f}'.format(pred_highlow_prob)
        print('[high] {}, {}, {} | {}, {}, {}'.format(digit_acc_msg,target_digit_prob_msg,pred_digit_prob_msg,\
                                                              high_acc_msg,target_high_prob_msg,pred_high_prob_msg))
    return None


def print_all_brief_stats(even_avg_stats,odd_avg_stats,low_avg_stats,high_avg_stats,pair_amount):
    print_each_brief_stats(even_avg_stats,pair_amount,0)
    print_each_brief_stats(odd_avg_stats,pair_amount,1)
    print_each_brief_stats(low_avg_stats,pair_amount,2)
    print_each_brief_stats(high_avg_stats,pair_amount,3)
    return None


def print_each_overall_stats(avg_stats,pair_amount,stats_amount=0):
    [digit_correct, parity_correct, highlow_correct, target_digit_prob, target_parity_prob, target_highlow_prob, \
     pred_digit_prob, pred_parity_prob, pred_highlow_prob, \
     nontarget_max_digit_prob, nontarget_max_parity_prob, nontarget_max_highlow_prob, \
     target_digit, target_parity, target_highlow, \
     pred_digit, pred_parity, pred_highlow, \
     nontarget_max_digit, nontarget_max_parity, nontarget_max_highlow] = avg_stats
    digit_correct_msg = 'Target Side: correct digit prediction: {:d}/{:d} pairs'.format(int(digit_correct*pair_amount),pair_amount)
    target_digit_prob_msg = 'Target Side: labeled digit certainty: {:.3f}%'.format(target_digit_prob*100)
    pred_digit_prob_msg = 'Target Side: predicted digit certainty: {:.3f}%'.format(pred_digit_prob*100)
    nontarget_max_digit_prob_msg = 'Nontarget Side: predicted digit certainty: {:.3f}%'.format(nontarget_max_digit_prob*100)
    if stats_amount == 0:
        even_correct_msg = 'correct even prediction: {:d}/{:d} pairs'.format(int(parity_correct*pair_amount),pair_amount)
        target_even_prob_msg = 'labeled even certainty: {:.3f}%'.format(target_parity_prob*100)
        pred_even_prob_msg = 'predicted even certainty: {:.3f}%'.format(pred_parity_prob*100)
        nontarget_max_even_prob_msg = 'predicted even certainty: {:.3f}%'.format(nontarget_max_parity_prob*100)
        print('[Statistics by using only even-goal-directed c-EB]: ')
        print('   {} | {}'.format(digit_correct_msg, even_correct_msg))
        print('   {} | {}'.format(target_digit_prob_msg, target_even_prob_msg))
        print('   {} | {}'.format(pred_digit_prob_msg, pred_even_prob_msg))
        print('   {} | {}'.format(nontarget_max_digit_prob_msg, nontarget_max_even_prob_msg))
    elif stats_amount == 1:
        odd_correct_msg = 'correct odd prediction: {:d}/{:d} pairs'.format(int(parity_correct*pair_amount),pair_amount)
        target_odd_prob_msg = 'labeled odd certainty: {:.3f}%'.format(target_parity_prob*100)
        pred_odd_prob_msg = 'predicted odd certainty: {:.3f}%'.format(pred_parity_prob*100)
        nontarget_max_odd_prob_msg = 'predicted odd certainty: {:.3f}%'.format(nontarget_max_parity_prob*100)
        print('[Statistics by using only odd-goal-directed c-EB]: ')
        print('   {} | {}'.format(digit_correct_msg, odd_correct_msg))
        print('   {} | {}'.format(target_digit_prob_msg, target_odd_prob_msg))
        print('   {} | {}'.format(pred_digit_prob_msg, pred_odd_prob_msg))
        print('   {} | {}'.format(nontarget_max_digit_prob_msg, nontarget_max_odd_prob_msg))
    elif stats_amount == 2:
        low_correct_msg = 'correct low prediction: {:d}/{:d} pairs'.format(int(highlow_correct*pair_amount),pair_amount)
        target_low_prob_msg = 'labeled low certainty: {:.3f}%'.format(target_highlow_prob*100)
        pred_low_prob_msg = 'predicted low certainty: {:.3f}%'.format(pred_highlow_prob*100)
        nontarget_max_low_prob_msg = 'predicted low certainty: {:.3f}%'.format(nontarget_max_highlow_prob*100)
        print('[Statistics by using only low-goal-directed c-EB]: ')
        print('   {} | {}'.format(digit_correct_msg, low_correct_msg))
        print('   {} | {}'.format(target_digit_prob_msg, target_low_prob_msg))
        print('   {} | {}'.format(pred_digit_prob_msg, pred_low_prob_msg))
        print('   {} | {}'.format(nontarget_max_digit_prob_msg, nontarget_max_low_prob_msg))
    elif stats_amount == 3:
        high_correct_msg = 'correct high prediction: {:d}/{:d} pairs'.format(int(highlow_correct*pair_amount),pair_amount)
        target_high_prob_msg = 'labeled high certainty: {:.3f}%'.format(target_highlow_prob*100)
        pred_high_prob_msg = 'predicted high certainty: {:.3f}%'.format(pred_highlow_prob*100)
        nontarget_max_high_prob_msg = 'predicted high certainty: {:.3f}%'.format(nontarget_max_highlow_prob*100)
        print('[Statistics by using only high-goal-directed c-EB]: ')
        print('   {} | {}'.format(digit_correct_msg, high_correct_msg))
        print('   {} | {}'.format(target_digit_prob_msg, target_high_prob_msg))
        print('   {} | {}'.format(pred_digit_prob_msg, pred_high_prob_msg))
        print('   {} | {}'.format(nontarget_max_digit_prob_msg, nontarget_max_high_prob_msg))
    return None


def print_all_overall_stats(even_avg_stats, odd_avg_stats, low_avg_stats, high_avg_stats, pair_amount):
    print_each_overall_stats(even_avg_stats,pair_amount,0)
    print('\n***********************************************************************************\n')
    print_each_overall_stats(odd_avg_stats,pair_amount,1)
    print('\n***********************************************************************************\n')
    print_each_overall_stats(low_avg_stats,pair_amount,2)
    print('\n***********************************************************************************\n')
    print_each_overall_stats(high_avg_stats,pair_amount,3)
    return None


def print_example_each_msg(stats,prob_0_1,pyes,isConcise=True):
    correctness = ["wrong", "correct"]
    parity = ["even","odd"]
    highlow = ["low","high"]
    digit_correct_msg = 'Target Side: digit prediction is {}'.format(correctness[stats[0].item()][:15])
    if prob_0_1 == 0:
        parity_correct_msg = 'even prediction is {}'.format(correctness[stats[1].item()][:15])
        highlow_correct_msg = 'low prediction is {}'.format(correctness[stats[2].item()][:15])
    else:
        parity_correct_msg = 'odd prediction is {}'.format(correctness[stats[1].item()][:15])
        highlow_correct_msg = 'high prediction is {}'.format(correctness[stats[2].item()][:15])
    
    target_digit_msg = 'Target Side: lbl digit: {:d}'.format(stats[12].item())
    target_parity_msg = 'lbl parity: {}'.format(parity[stats[13]][:15])
    target_highlow_msg = 'lbl highlow: {}'.format(highlow[stats[14]][:15])
    target_digit_prob_msg = 'Target Side: lbl digit certainty: {:.3f}%'.format(stats[3].item()*100)
    target_parity_prob_msg = 'lbl parity certainty: {:.3f}%'.format(stats[4].item()*100)
    target_highlow_prob_msg = 'lbl high/low certainty: {:.3f}%'.format(stats[5].item()*100)
    
    predicted_digit_msg = 'Target Side: pred digit: {:d}'.format(stats[15].item())
    predicted_parity_msg = 'pred parity: {}'.format(parity[stats[16]][:15])
    predicted_highlow_msg = 'pred highlow: {}'.format(highlow[stats[17]][:15])
    predicted_digit_prob_msg = 'Target Side: pred digit certainty: {:.3f}%'.format(stats[6].item()*100)
    predicted_parity_prob_msg = 'pred parity certainty: {:.3f}%'.format(stats[7].item()*100)
    predicted_highlow_prob_msg = 'pred high/low certainty: {:.3f}%'.format(stats[8].item()*100)
    
    nontarget_max_digit_msg = 'Nontarget Side: pred digit: {:d}'.format(stats[18].item())
    nontarget_max_parity_msg = 'pred parity: {}'.format(parity[stats[19]][:15])
    nontarget_max_highlow_msg = 'pred highlow: {}'.format(highlow[stats[20]][:15])
    nontarget_max_digit_prob_msg = 'Nontarget Side: pred digit certainty: {:.3f}%'.format(stats[9].item()*100)
    nontarget_max_parity_prob_msg = 'pred parity certainty: {:.3f}%'.format(stats[10].item()*100)
    nontarget_max_highlow_prob_msg = 'pred high/low certainty: {:.3f}%'.format(stats[11].item()*100)
    
    if not isConcise:
        all_msg = [digit_correct_msg, parity_correct_msg, highlow_correct_msg, \
                   target_digit_msg, target_parity_msg, target_highlow_msg, \
                   target_digit_prob_msg, target_parity_prob_msg, target_highlow_prob_msg, \
                   predicted_digit_msg, predicted_parity_msg, predicted_highlow_msg, \
                   predicted_digit_prob_msg, predicted_parity_prob_msg, predicted_highlow_prob_msg, \
                   nontarget_max_digit_msg, nontarget_max_parity_msg, nontarget_max_highlow_msg, \
                   nontarget_max_digit_prob_msg, nontarget_max_parity_prob_msg, nontarget_max_highlow_prob_msg]
    else:
        if pyes:
            all_msg = [digit_correct_msg, parity_correct_msg, \
                       target_digit_msg, target_parity, \
                       target_digit_prob_msg, target_parity_prob_msg, \
                       predicted_digit_msg, predicted_parity_msg, \
                       predicted_digit_prob_msg, predicted_parity_prob_msg, \
                       nontarget_max_digit_msg, nontarget_max_parity_msg, \
                       nontarget_max_digit_prob_msg, nontarget_max_parity_prob_msg]
        else:
            all_msg = [digit_correct_msg, highlow_correct_msg, \
                       target_digit_msg, target_highlow_msg, \
                       target_digit_prob_msg, target_highlow_prob_msg, \
                       predicted_digit_msg, predicted_highlow_msg, \
                       predicted_digit_prob_msg, predicted_highlow_prob_msg, \
                       nontarget_max_digit_msg, nontarget_max_highlow_msg, \
                       nontarget_max_digit_prob_msg, nontarget_max_highlow_prob_msg]
    
    return all_msg


def show_cEB_example(all_X,all_y,all_p,all_hl,model,data_ind,eg_ind,isConcise=False):
    parity = ["even","odd"]
    highlow = ["low","high"]
    inputs = all_X[data_ind].reshape(1,-1)
    y = all_y[data_ind]
    p = all_p[data_ind]
    hl = all_hl[data_ind]
    
    pair_amount = all_X.shape[0]
    
    true_id_0 = y.data[0]
    true_id_1 = y.data[1]
    true_p_0 = p.data[0]
    true_p_1 = p.data[1]
    true_hl_0 = hl.data[0]
    true_hl_1 = hl.data[1]

    inputs_plot = torch.cat((inputs[:,0:28*28].view(28,28),inputs[:,28*28:28*56].view(28,28)),1).cpu()
    
    true_img_even, true_img_odd, even_stats, odd_stats = obtain_example_stats(inputs,y,p,hl,model,True)
    true_img_low, true_img_high, low_stats, high_stats = obtain_example_stats(inputs,y,p,hl,model,False)
    
    even_msg = print_example_each_msg(even_stats,0,True,isConcise=isConcise)
    odd_msg = print_example_each_msg(odd_stats,1,True,isConcise=isConcise)
    low_msg = print_example_each_msg(low_stats,0,False,isConcise=isConcise)
    high_msg = print_example_each_msg(high_stats,1,False,isConcise=isConcise)

    s = 3
    f = plt.figure(figsize=[s*5,s*2.4])
    plt.subplot(2,3,1)
    plt.title('Example ({}) Input image: "{}" ({}), "{}" ({})'.format(eg_ind,true_id_0,parity[true_p_0][:15],\
                                                                      true_id_1,parity[true_p_1][:15]))
    plt.imshow(inputs_plot.numpy(), cmap='gray')

    plt.subplot(2,3,2)
    plt.title('c-EB generated input: even goal only')
    plt.imshow(true_img_even, cmap='gray')

    plt.subplot(2,3,3)
    plt.title('c-EB generated input: odd goal only')
    plt.imshow(true_img_odd, cmap='gray')
    
    plt.subplot(2,3,4)
    plt.title('Example ({}) Input image: "{}" ({}), "{}" ({})'.format(eg_ind,true_id_0,highlow[true_hl_0][:15],\
                                                                      true_id_1,highlow[true_hl_1][:15]))
    plt.imshow(inputs_plot.numpy(), cmap='gray')

    plt.subplot(2,3,5)
    plt.title('c-EB generated input: low-value goal only')
    plt.imshow(true_img_low, cmap='gray')

    plt.subplot(2,3,6)
    plt.title('c-EB generated input: high-value goal only')
    plt.imshow(true_img_high, cmap='gray')
    plt.show() ; 
    
    if not isConcise:
        print('{The Even Goal Result}: ')
        print('   {} | {} | {}'.format(even_msg[0],even_msg[1],even_msg[2]))
        print('   {} | {} | {}'.format(even_msg[3],even_msg[4],even_msg[5]))
        print('   {} | {} | {}'.format(even_msg[6],even_msg[7],even_msg[8]))
        print('   {} | {} | {}'.format(even_msg[9],even_msg[10],even_msg[11]))
        print('   {} | {} | {}'.format(even_msg[12],even_msg[13],even_msg[14]))
        print('   {} | {} | {}'.format(even_msg[15],even_msg[16],even_msg[17]))
        print('   {} | {} | {}'.format(even_msg[18],even_msg[19],even_msg[20]))
        
        print('\n{The Odd Goal Result}: ')
        print('   {} | {} | {}'.format(odd_msg[0],odd_msg[1],odd_msg[2]))
        print('   {} | {} | {}'.format(odd_msg[3],odd_msg[4],odd_msg[5]))
        print('   {} | {} | {}'.format(odd_msg[6],odd_msg[7],odd_msg[8]))
        print('   {} | {} | {}'.format(odd_msg[9],odd_msg[10],odd_msg[11]))
        print('   {} | {} | {}'.format(odd_msg[12],odd_msg[13],odd_msg[14]))
        print('   {} | {} | {}'.format(odd_msg[15],odd_msg[16],odd_msg[17]))
        print('   {} | {} | {}'.format(odd_msg[18],odd_msg[19],odd_msg[20]))
        
        print('\n{The Low Goal Result}: ')
        print('   {} | {} | {}'.format(low_msg[0],low_msg[1],low_msg[2]))
        print('   {} | {} | {}'.format(low_msg[3],low_msg[4],low_msg[5]))
        print('   {} | {} | {}'.format(low_msg[6],low_msg[7],low_msg[8]))
        print('   {} | {} | {}'.format(low_msg[9],low_msg[10],low_msg[11]))
        print('   {} | {} | {}'.format(low_msg[12],low_msg[13],low_msg[14]))
        print('   {} | {} | {}'.format(low_msg[15],low_msg[16],low_msg[17]))
        print('   {} | {} | {}'.format(low_msg[18],low_msg[19],low_msg[20]))
        
        print('\n{The High Goal Result}: ')
        print('   {} | {} | {}'.format(high_msg[0],high_msg[1],high_msg[2]))
        print('   {} | {} | {}'.format(high_msg[3],high_msg[4],high_msg[5]))
        print('   {} | {} | {}'.format(high_msg[6],high_msg[7],high_msg[8]))
        print('   {} | {} | {}'.format(high_msg[9],high_msg[10],high_msg[11]))
        print('   {} | {} | {}'.format(high_msg[12],high_msg[13],high_msg[14]))
        print('   {} | {} | {}'.format(high_msg[15],high_msg[16],high_msg[17]))
        print('   {} | {} | {}'.format(high_msg[18],high_msg[19],high_msg[20]))
        
    else:
        print('{The Even Goal Result}: ')
        print('   {} | {}'.format(even_msg[0],even_msg[1]))
        print('   {} | {}'.format(even_msg[2],even_msg[3]))
        print('   {} | {}'.format(even_msg[4],even_msg[5]))
        print('   {} | {}'.format(even_msg[6],even_msg[7]))
        print('   {} | {}'.format(even_msg[8],even_msg[9]))
        print('   {} | {}'.format(even_msg[10],even_msg[11]))
        print('   {} | {}'.format(even_msg[12],even_msg[13]))
        
        print('\n{The Odd Goal Result}: ')
        print('   {} | {}'.format(odd_msg[0],odd_msg[1]))
        print('   {} | {}'.format(odd_msg[2],odd_msg[3]))
        print('   {} | {}'.format(odd_msg[4],odd_msg[5]))
        print('   {} | {}'.format(odd_msg[6],odd_msg[7]))
        print('   {} | {}'.format(odd_msg[8],odd_msg[9]))
        print('   {} | {}'.format(odd_msg[10],odd_msg[11]))
        print('   {} | {}'.format(odd_msg[12],odd_msg[13]))
        
        print('\n{The Low Goal Result}: ')
        print('   {} | {}'.format(low_msg[0],low_msg[1]))
        print('   {} | {}'.format(low_msg[2],low_msg[3]))
        print('   {} | {}'.format(low_msg[4],low_msg[5]))
        print('   {} | {}'.format(low_msg[6],low_msg[7]))
        print('   {} | {}'.format(low_msg[8],low_msg[9]))
        print('   {} | {}'.format(low_msg[10],low_msg[11]))
        print('   {} | {}'.format(low_msg[12],low_msg[13]))
        
        print('\n{The High Goal Result}: ')
        print('   {} | {}'.format(high_msg[0],high_msg[1]))
        print('   {} | {}'.format(high_msg[2],high_msg[3]))
        print('   {} | {}'.format(high_msg[4],high_msg[5]))
        print('   {} | {}'.format(high_msg[6],high_msg[7]))
        print('   {} | {}'.format(high_msg[8],high_msg[9]))
        print('   {} | {}'.format(high_msg[10],high_msg[11]))
        print('   {} | {}'.format(high_msg[12],high_msg[13]))
    
    return None

