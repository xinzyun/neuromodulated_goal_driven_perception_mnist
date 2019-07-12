### Train the classifier and make small amount of test using the currently trained model to evaluate the training progress
### This code script was completely written by us, who are anonymous authors of the submitted paper under review.

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, copy ; sys.path.append('..')
from define_model import MnistClassifier
from load_mnist_data_pair import Dataloader
from predict_w_cEB import get_each_ceb_accuracy
from show_cEB_results import print_all_brief_stats

reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()

if torch.cuda.is_available():
    useCuda = True
else:
    useCuda = False
    
def train_model(total_steps = 4200,test_every = 200,test_pair_num = 2000, toTest = True):
    if useCuda:
        model = MnistClassifier().cuda()
    else:
        model = MnistClassifier()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    dataloader = Dataloader(512) 
    
    global_step = 0; 
    running_loss = 0; 
    loss_hist = []
    
    print('Start to train {} steps, each with {} all-combination pairs.'.format(total_steps,256))
    if toTest:
        print('Test every {} steps with {} opposite-property pairs.'.format(test_every,test_pair_num))
        print('========================================================================')
    
    # generic train loop
    for global_step in range(global_step, total_steps+global_step+1):
        
        X, y, p, hl = dataloader.next()
        X = X.view(-1,28*56)
        y = y.view(-1,2)
        p = p.view(-1,2)
        hl = hl.view(-1,2)
        logits = model(X)
        y0_hat = F.log_softmax(torch.mean(torch.cat((logits[0][:,0:10].unsqueeze_(2),logits[2][:,0:10].unsqueeze_(2)),2),2), dim=-1)
        y1_hat = F.log_softmax(torch.mean(torch.cat((logits[0][:,10:20].unsqueeze_(2),logits[2][:,10:20].unsqueeze_(2)),2),2), dim=-1)
        p0_hat = F.log_softmax(logits[1][:,0:2], dim=-1)
        p1_hat = F.log_softmax(logits[1][:,2:4], dim=-1)
        hl0_hat = F.log_softmax(logits[3][:,0:2], dim=-1)
        hl1_hat = F.log_softmax(logits[3][:,2:4], dim=-1)
        loss_y0 = F.nll_loss(y0_hat, y[:,0])
        loss_y1 = F.nll_loss(y1_hat, y[:,1])
        loss_p0 = F.nll_loss(p0_hat, p[:,0]) 
        loss_p1 = F.nll_loss(p1_hat, p[:,1])
        loss_hl0 = F.nll_loss(hl0_hat, hl[:,0]) 
        loss_hl1 = F.nll_loss(hl1_hat, hl[:,1])
        loss = loss_y0 + loss_y1 + loss_p0 + loss_p1 + loss_hl0 + loss_hl1
        loss.backward() ; optimizer.step() ; optimizer.zero_grad()
        
        loss_cpu = loss
        loss_cpu = loss_cpu.cpu()
        np_loss = loss_cpu.data.numpy()
        running_loss = np_loss if running_loss is None else .99*running_loss + (1-.99)*np_loss
        loss_hist.append(running_loss)
        
        # ======== TEST THE TRAINING PROGRESS ======== #
        if (global_step % test_every == 0) and toTest:
            model_test = copy.deepcopy(model)
            even_avg_stats, odd_avg_stats, low_avg_stats, high_avg_stats, pair_amount, all_X, all_y, all_p, all_hl = get_each_ceb_accuracy(model_test,dataloader,pair_num=test_pair_num)
            print('step {}/{} | loss: {:.4f}'.format(global_step, total_steps, running_loss))
            print_all_brief_stats(even_avg_stats,odd_avg_stats,low_avg_stats,high_avg_stats,pair_amount)
            print('--------------------------------------------------------------------')
        
    return model, dataloader
