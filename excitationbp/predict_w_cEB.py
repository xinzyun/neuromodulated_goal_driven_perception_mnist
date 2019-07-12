### Prepare for digit and goal prediction with the goal-related pixels in the noisy MNIST pair highlighted by c-EB, driven by the goal.
### This code script was completely written by us, who are anonymous authors of the submitted paper under review.

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, sys, copy ; sys.path.append('..')
import excitationbp as eb
import random

reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()

if torch.cuda.is_available():
    useCuda = True
else:
    useCuda = False
    

def select_inputs_once(dataloader):
    X, y, p, hl = dataloader.next(mode='test')
    X = X.view(-1,28*56)
    y = y.view(-1,2)
    p = p.view(-1,2)
    hl = hl.view(-1,2)
    keep_ind = []
    for i in range(X.shape[0]):
        if ((p[i]==0).nonzero().shape[0]==1) and ((hl[i]==0).nonzero().shape[0]==1):
            keep_ind.append(i)
    new_X = X[keep_ind]
    new_y = y[keep_ind]
    new_p = p[keep_ind]
    new_hl = hl[keep_ind]
    
    return new_X, new_y, new_p, new_hl


def select_all_cEB_inputs(dataloader,pair_num=10000):
    all_X, all_y, all_p, all_hl = select_inputs_once(dataloader)
    for i in range(pair_num):
        new_X, new_y, new_p, new_hl = select_inputs_once(dataloader)
        all_X = torch.cat((all_X, new_X),0)
        all_y = torch.cat((all_y, new_y),0)
        all_p = torch.cat((all_p, new_p),0)
        all_hl = torch.cat((all_hl, new_hl),0)
        if all_X.shape[0] >= pair_num:
            break
    all_X = all_X[0:pair_num]
    all_y = all_y[0:pair_num]
    all_p = all_p[0:pair_num]
    all_hl = all_hl[0:pair_num]
    
    return all_X, all_y, all_p, all_hl


def obtain_ceb_prob_inputs(inputs,y,p,hl,model,pyes=True):
    eb.use_eb(True)
    true_id_0 = y.data[0]
    true_id_1 = y.data[1]
    parity = ["even","odd"]
    true_p_0 = p.data[0]
    true_p_1 = p.data[1]
    highlow = ["low","high"]
    true_hl_0 = hl.data[0]
    true_hl_1 = hl.data[1]
    
    prob_outputs_true_p_even = Variable(torch.zeros(1,4)) ; 
    prob_outputs_true_p_even.data[:,0] += 1; prob_outputs_true_p_even.data[:,2] += 1; 
    
    prob_outputs_true_p_odd = Variable(torch.zeros(1,4)) ; 
    prob_outputs_true_p_odd.data[:,1] += 1; prob_outputs_true_p_odd.data[:,3] += 1; 
    
    prob_outputs_true_hl_low = Variable(torch.zeros(1,4)) ; 
    prob_outputs_true_hl_low.data[:,0] += 1; prob_outputs_true_hl_low.data[:,2] += 1; 
    
    prob_outputs_true_hl_high = Variable(torch.zeros(1,4)) ; 
    prob_outputs_true_hl_high.data[:,1] += 1; prob_outputs_true_hl_high.data[:,3] += 1; 
    
    if pyes:
        prob_outputs_true_0 = prob_outputs_true_p_even
        prob_outputs_true_1 = prob_outputs_true_p_odd
        top_layer_num = -3
    else:
        prob_outputs_true_0 = prob_outputs_true_hl_low
        prob_outputs_true_1 = prob_outputs_true_hl_high
        top_layer_num = -1
    
    if useCuda:
        prob_outputs_true_0 = prob_outputs_true_0.type(torch.cuda.FloatTensor)
        prob_outputs_true_1 = prob_outputs_true_1.type(torch.cuda.FloatTensor)
    
    prob_inputs_true_0 = eb.excitation_backprop(model, inputs, prob_outputs_true_0, contrastive=True, top_layer=top_layer_num)
    prob_inputs_true_1 = eb.excitation_backprop(model, inputs, prob_outputs_true_1, contrastive=True, top_layer=top_layer_num)
    
    if useCuda:
        prob_inputs_true_0 = prob_inputs_true_0.type(torch.cuda.FloatTensor)
        prob_inputs_true_1 = prob_inputs_true_1.type(torch.cuda.FloatTensor)
    
    return prob_inputs_true_0, prob_inputs_true_1


def ceb_pred_accuracy(prob_input,prob_0_1,y,p,hl,model,pyes=True):
    eb.use_eb(False)
    ceb_input = (prob_input).clamp(min=0)
    newX = Variable(ceb_input/ceb_input.max(1)[0]*1.7)
    if useCuda:
        newX = newX.type(torch.cuda.FloatTensor)
        model = model.cuda()
    logits = model(newX)
    
    if pyes:
        target_ind = (p==prob_0_1).nonzero() # 0 or 1
        target_digit = y[p==prob_0_1]  #0~9
        target_parity = prob_0_1
        target_highlow = hl[p==prob_0_1]
    else:
        target_ind = (hl==prob_0_1).nonzero() # 0 or 1
        target_digit = y[hl==prob_0_1]  #0~9
        target_parity = p[hl==prob_0_1]
        target_highlow = prob_0_1
    
    if pyes:
        target_y_hat = F.softmax(logits[0][:,(0+target_ind*10):(10+target_ind*10)], dim=-1)
    else:
        target_y_hat = F.softmax(logits[2][:,(0+target_ind*10):(10+target_ind*10)], dim=-1)
    #target_y_hat = F.softmax(logits[0][:,(0+target_ind*10):(10+target_ind*10)], dim=-1)
    target_digit_prob = target_y_hat[:,target_digit].data
    
    target_p_hat = F.softmax(logits[1][:,(0+target_ind*2):(2+target_ind*2)], dim=-1)
    target_parity_prob = target_p_hat[:,target_parity].data
    
    target_hl_hat = F.softmax(logits[3][:,(0+target_ind*2):(2+target_ind*2)], dim=-1)
    target_highlow_prob = target_hl_hat[:,target_highlow].data
    
    pred_digit = target_y_hat.data.max(1)[1]
    pred_digit_prob = target_y_hat.data.max(1)[0]
    
    pred_parity = target_p_hat.data.max(1)[1]
    pred_parity_prob = target_p_hat.data.max(1)[0]
    
    pred_highlow = target_hl_hat.data.max(1)[1]
    pred_highlow_prob = target_hl_hat.data.max(1)[0]
    
    digit_correct = Variable(target_digit == pred_digit).type(torch.LongTensor).data
    parity_correct = Variable(target_parity == pred_parity).type(torch.LongTensor).data
    highlow_correct = Variable(target_highlow == pred_highlow).type(torch.LongTensor).data
    
    nontarget_ind = 1 - target_ind
    if pyes:
        nontarget_y_hat = F.softmax(logits[0][:,(0+nontarget_ind*10):(10+nontarget_ind*10)], dim=-1)
    else:
        nontarget_y_hat = F.softmax(logits[2][:,(0+nontarget_ind*10):(10+nontarget_ind*10)], dim=-1)
    #nontarget_y_hat = F.softmax(logits[0][:,(0+nontarget_ind*10):(10+nontarget_ind*10)], dim=-1)
    nontarget_max_digit = nontarget_y_hat.data.max(1)[1]
    nontarget_max_digit_prob = nontarget_y_hat.data.max(1)[0]
    
    nontarget_p_hat = F.softmax(logits[1][:,(0+nontarget_ind*2):(2+nontarget_ind*2)], dim=-1)
    nontarget_max_parity = nontarget_p_hat.data.max(1)[1]
    nontarget_max_parity_prob = nontarget_p_hat.data.max(1)[0]
    
    nontarget_hl_hat = F.softmax(logits[3][:,(0+nontarget_ind*2):(2+nontarget_ind*2)], dim=-1)
    nontarget_max_highlow = nontarget_hl_hat.data.max(1)[1]
    nontarget_max_highlow_prob = nontarget_hl_hat.data.max(1)[0]
    
    return_list = [digit_correct, parity_correct, highlow_correct, target_digit_prob, target_parity_prob, target_highlow_prob, \
                   pred_digit_prob, pred_parity_prob, pred_highlow_prob, \
                   nontarget_max_digit_prob, nontarget_max_parity_prob, nontarget_max_highlow_prob, \
                   target_digit, target_parity, target_highlow, \
                   pred_digit, pred_parity, pred_highlow, \
                   nontarget_max_digit, nontarget_max_parity, nontarget_max_highlow]
    
    return return_list


def get_zero_one_accuracy(inputs,y,p,hl,model,pyes=True):
    model_copy = copy.deepcopy(model)
    prob_inputs_true_0, prob_inputs_true_1 = obtain_ceb_prob_inputs(inputs,y,p,hl,model_copy,pyes)
    model_copy = copy.deepcopy(model)
    zero_stats = ceb_pred_accuracy(prob_inputs_true_0,0,y,p,hl,model_copy,pyes)
    model_copy = copy.deepcopy(model)
    one_stats = ceb_pred_accuracy(prob_inputs_true_1,1,y,p,hl,model_copy,pyes)
    combo_stats = np.mean(np.asarray([zero_stats,one_stats]),0).tolist()
    
    return combo_stats, zero_stats, one_stats


def get_each_ceb_accuracy(model,dataloader,pair_num=10000):
    all_X, all_y, all_p, all_hl = select_all_cEB_inputs(dataloader,pair_num)
    all_even_stats = []
    all_odd_stats = []
    all_low_stats = []
    all_high_stats = []
    pair_amount = all_X.shape[0]
    for i in range(pair_amount):
        inputs = all_X[i].reshape(1,-1)
        y = all_y[i]
        p = all_p[i]
        hl = all_hl[i]
        _, even_stats, odd_stats = get_zero_one_accuracy(inputs,y,p,hl,model,True)
        _, low_stats, high_stats = get_zero_one_accuracy(inputs,y,p,hl,model,False)
        all_even_stats.append(even_stats)
        all_odd_stats.append(odd_stats)
        all_low_stats.append(low_stats)
        all_high_stats.append(high_stats)

    even_avg_stats = np.mean(np.asarray(all_even_stats),0)
    even_avg_stats = even_avg_stats.tolist()
    
    odd_avg_stats = np.mean(np.asarray(all_odd_stats),0)
    odd_avg_stats = odd_avg_stats.tolist()
    
    low_avg_stats = np.mean(np.asarray(all_low_stats),0)
    low_avg_stats = low_avg_stats.tolist()
    
    high_avg_stats = np.mean(np.asarray(all_high_stats),0)
    high_avg_stats = high_avg_stats.tolist()
    
    return even_avg_stats, odd_avg_stats, low_avg_stats, high_avg_stats, pair_amount, all_X, all_y, all_p, all_hl


def obtain_example_stats(inputs,y,p,hl,model,pyes):
    model_copy = copy.deepcopy(model)
    prob_inputs_true_0, prob_inputs_true_1 = obtain_ceb_prob_inputs(inputs,y,p,hl,model_copy,pyes)
    model_copy = copy.deepcopy(model)
    zero_stats = ceb_pred_accuracy(prob_inputs_true_0,0,y,p,hl,model_copy,pyes)
    model_copy = copy.deepcopy(model)
    one_stats = ceb_pred_accuracy(prob_inputs_true_1,1,y,p,hl,model_copy,pyes)
    
    true_img_0 = torch.cat((prob_inputs_true_0[:,0:28*28].view(28,28),\
                            prob_inputs_true_0[:,28*28:28*56].view(28,28)),1).clamp(min=0).cpu().numpy()
    true_img_1 = torch.cat((prob_inputs_true_1[:,0:28*28].view(28,28),\
                            prob_inputs_true_1[:,28*28:28*56].view(28,28)),1).clamp(min=0).cpu().numpy()
    
    return true_img_0, true_img_1, zero_stats, one_stats


def obtain_example_labels(inputs,prob_0_1,y,p,hl,pyes):
    if pyes:
        target_ind = (p==prob_0_1).nonzero() # 0 or 1
        target_digit = y[p==prob_0_1]  #0~9
        target_parity = prob_0_1
        target_highlow = hl[p==prob_0_1]
    else:
        target_ind = (hl==prob_0_1).nonzero() # 0 or 1
        target_digit = y[hl==prob_0_1]  #0~9
        target_parity = p[hl==prob_0_1]
        target_highlow = prob_0_1
    
    labels = [target_ind,target_digit,target_parity,target_highlow]
    
    return labels


