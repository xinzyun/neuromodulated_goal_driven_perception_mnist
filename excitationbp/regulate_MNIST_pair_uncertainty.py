### Describe the ACh and NE neuromodulation process to make robust goal prediction in uncertain domains
### This code script was completely written by us, who are anonymous authors of the submitted paper under review.

import numpy as np
import random, copy
import matplotlib.pyplot as plt
from predict_w_cEB import select_all_cEB_inputs, obtain_example_stats, obtain_example_labels

def MNIST_action_select(m,beta):
    prob = np.zeros(len(m));
    
    # get softmax probabilities
    for i in range(len(m)):
        prob[i] = np.exp(beta*m[i])/np.sum(np.exp(beta*m));
    
    r = random.random(); # random number for action selection
    sumprob = 0;
    
    # choose an action based on the probability distribution
    i = 0;
    done = False;
    
    while (not done) and (i < len(m)):
        sumprob += prob[i];
        if sumprob >= r:
            act = i;
            done = True;
        else:
            i += 1;
    
    return act, prob


def MNIST_digit_predict(inputs,y,p,hl,model,act):
    pyes = (act < 2)
    _, _, zero_stats, one_stats = obtain_example_stats(inputs,y,p,hl,model,pyes)
    
    if act % 2 == 0:
        predicted_digit = zero_stats[15].item()
    else:
        predicted_digit = one_stats[15].item()
        
    return predicted_digit


def MNIST_digit_label(inputs,y,p,hl,cue):
    pyes = (cue < 2)
    prob_0_1 = int(cue % 2 != 0)
    
    cued_labels = obtain_example_labels(inputs,prob_0_1,y,p,hl,pyes)
    cued_digit = cued_labels[1].item()
        
    return cued_digit


def MNIST_choice_trend(choice,num_considered=10,num_threshold=8):
    trend_choice = np.zeros(len(choice),dtype=int)-1;
    for i in range(len(choice)):
        considered = choice[max(i+1-num_considered,0):i+1]
        (values,counts) = np.unique(considered,return_counts=True)
        ind=np.argmax(counts)
        if counts[ind] >= num_threshold and values[ind]==considered[-1]:
            trend_choice[i] = values[ind]
        
    return trend_choice


def MNIST_trend_lag(trend_choice,major_cues,trial_intervals):
    lag_trend = np.zeros(len(major_cues),dtype=int)
    ind = 0
    sum_lag = 0
    avg_count = 0
    for i in range(len(major_cues)):
        t = 0
        done = False
        while (not done) and t < trial_intervals[i]:
            if trend_choice[ind+t] == major_cues[i]:
                lag_trend[i]=t
                if i > 0 and major_cues[i-1] != major_cues[i]:
                    sum_lag += t
                    avg_count += 1
                done = True
            t += 1
        ind += trial_intervals[i]
    
    avg_lag = sum_lag/avg_count
    return lag_trend,avg_lag


def MNIST_uncertainty_task(model,dataloader,ACH_CORRECT=1.04,ACH_INCORRECT=0.99,NE_INCORRECT=1.02,NE_CORRECT=0.97,trialRange=0, trialInterval = 200,alterValid=False,validity_choices=[0.99,0.85,0.70],previous_intervals=None, previous_cues=None, num_considered=10,num_threshold=8,maxACh = 10,hasLimit=True,hasAChLesion=False,hasNELesion=False, beta=1):
    #beta = 1;
    
    #Rates for ACh and NE. ACh should grow with each correct choice. NE should 
    # grow with each incorrect choice.  May want to adjust these
    #ACH_CORRECT = 1.04;
    #ACH_INCORRECT = 0.99;
    #NE_INCORRECT = 1.02; 
    #NE_CORRECT = 0.97;
    
    #When NE goes above the threshold level, it causes a network reset. See
    # theory by Bouret and Sara in 2005.
    NE_RESET = 0.25;
    maxNE = 1.0;
    
    #Capped the ACh level, otherwise it can get too high and break the softmax
    # function.
    #maxACh = 10;
    
    #ACh is stimulus specific. So, made an array of ACh neurons equal to the
    # number of cues. NE is more of a broadcast signal. So, only one NE neuron.
    numCues = 4;
    ACh = np.ones(numCues);
    NE = NE_RESET;
    
    # numSwitches is how many different cues are chosen
    numSwitches = 10;
    
    # trialInterval is the number of trials for each cue.
    #trialInterval = 200;
    
    # arrays used for plotting purposes
    correct = np.zeros(int(numSwitches*(trialInterval+trialRange)),dtype=int);
    ach_level = np.zeros((int(numSwitches*(trialInterval+trialRange)),numCues),dtype=float);
    ach_avg_level = np.zeros(int(numSwitches*(trialInterval+trialRange)),dtype=float);
    ne_level = np.zeros(int(numSwitches*(trialInterval+trialRange)),dtype=float);
    choice = np.zeros(int(numSwitches*(trialInterval+trialRange)),dtype=int);
    correct_lbl = np.zeros(int(numSwitches*(trialInterval+trialRange)),dtype=int);
    
    # obtain testing pairs
    all_X, all_y, all_p, all_hl = select_all_cEB_inputs(dataloader,int(numSwitches*(trialInterval+trialRange)))
    
    # choices of cue validity
    #validity_choices = [0.99, 0.85, 0.70];
    
    # choices of cue labels
    cue_labels = ['even','odd','low','high'];
    
    print('A cue was randomly picked among integers 0~3 (i.e. even, odd, low, high respectively) every {:d}+/-{:d} trials for {:d} times.'.format(trialInterval, trialRange, numSwitches));
    print('The cue validity was randomly selected among ' + str(validity_choices) + ' in each trial.');
    if alterValid:
        print('A major cue or its alternative is possible for each trial depends on the cue validity.')
    #print('')
    
    inx = 0;
    times_match_but_invalid = 0;
    times_pred_not_match_cue = 0;
    times_pred_not_match_guess = 0;
    times_guess_not_match_cue = 0;
    
    times_match_cue1_valid = 0;
    times_match_cue2_valid = 0;
    
    times_match_cue1_invalid_alter = 0;
    
    trial_intervals = np.zeros(int(numSwitches),dtype=int);
    major_cues = np.zeros(int(numSwitches),dtype=int);
    acts_prob = np.zeros((int(numSwitches*(trialInterval+trialRange)),numCues),dtype=float);
    
    for i in range(numSwitches):
        if previous_cues is None:
            cue = random.randint(0,numCues-1); # pick a cue at random
        else:
            cue = previous_cues[i]
        #print(cue)
        major_cues[i] = cue
        
        if (type(validity_choices) is int) or (type(validity_choices) is float):
            cue_validity = validity_choices
        else:
            cue_validity = validity_choices[random.randint(0,len(validity_choices)-1)];
        
        if cue % 2 == 0:
            alter_cue = cue + 1
        else:
            alter_cue = cue - 1
        
        if trialRange == 0:
            temp_trialInterval = trialInterval
        else:
            if previous_intervals is None:
                temp_trialInterval = random.randint(trialInterval-trialRange,trialInterval+trialRange)
            else:
                temp_trialInterval = previous_intervals[i]
        
        trial_intervals[i] = temp_trialInterval
        
        '''
        print('Switch {:d}/{:d}: cue is {:d}:{}, cue validity is {:.2f}, and trial interval is {:d}.'.format(i+1,numSwitches,cue,cue_labels[cue],cue_validity,temp_trialInterval));
        if alterValid:
            print('                  altered cue is {:d}:{} with validity {:.2f}.'.format(alter_cue,cue_labels[alter_cue],1-cue_validity));
        '''
        
        for t in range(temp_trialInterval):
            inputs = all_X[inx].reshape(1,-1)
            y = all_y[inx]
            p = all_p[inx]
            hl = all_hl[inx]
            
            r = random.random(); # random number for cue validity selection
            if not alterValid:
                temp_cue = cue
            else:
                if r < cue_validity:
                    temp_cue = cue
                else:
                    temp_cue = alter_cue
            
            # Choose an action based on softmax function.
            # Choice doesn't have to be an array. But its convenient for
            # plotting purposes.
            temp_choice, actsProb = MNIST_action_select(ACh, beta);
            choice[inx] = temp_choice;
            
            # predict digit based on c-EB directed by temp_choice
            model_copy = copy.deepcopy(model)
            temp_predicted_digit = int(MNIST_digit_predict(inputs,y,p,hl,model_copy,temp_choice)); 
            # obtain guess-target digit 
            temp_guess_target_digit = int(MNIST_digit_label(inputs,y,p,hl,temp_choice));
            # obtain cued digit 
            temp_cued_digit = int(MNIST_digit_label(inputs,y,p,hl,temp_cue));
            # obtain major cued digit
            cued_digit = int(MNIST_digit_label(inputs,y,p,hl,cue));
            
            predIsCorrect = (temp_predicted_digit == temp_cued_digit) and (temp_choice == temp_cue);
            predMatchGuess = (temp_predicted_digit == temp_guess_target_digit);
            guessIsCorrect = (temp_guess_target_digit == temp_cued_digit) and (temp_choice == temp_cue);
            
            # If the predicted digit matches the cue && a random number is smaller than the random cue validity
            #   increase ACh level; check for max
            #   decrease NE level
            if ((not alterValid) and predIsCorrect and r < cue_validity) or (alterValid and predIsCorrect):
                if not hasAChLesion:
                    ACh[temp_choice] = ACh[temp_choice] * ACH_CORRECT;
                    ACh[temp_choice] = min(maxACh,ACh[temp_choice]);
                if not hasNELesion:
                    NE = NE * NE_CORRECT;
                    if hasLimit:
                        NE = max(NE_RESET,NE);
            # Else incorrect choice
            #   decrease ACh level
            #   increase NE level
            else:
                if not hasAChLesion:
                    ACh[temp_choice] = ACh[temp_choice] * ACH_INCORRECT;
                if not hasNELesion:
                    NE = NE * NE_INCORRECT;
                    if hasLimit:
                        NE = min(maxNE,NE);
            
            if ((not alterValid) and predIsCorrect and r >= cue_validity):
                times_match_but_invalid += 1;
            if not predIsCorrect:
                times_pred_not_match_cue += 1;
            if not predMatchGuess:
                times_pred_not_match_guess += 1;
            if not guessIsCorrect:
                times_guess_not_match_cue += 1;
            
            if predIsCorrect and r < cue_validity:
                times_match_cue1_valid += 1;
            if alterValid and predIsCorrect and r >= cue_validity:
                times_match_cue2_valid += 1;
            if alterValid and r >= cue_validity and (temp_predicted_digit == cued_digit) and (temp_choice == cue):
                times_match_cue1_invalid_alter += 1;
            
            AChLevel = np.sum(ACh)/numCues;
            
            # Save values for plotting
            ach_level[inx,:] = ACh;
            ach_avg_level[inx] = AChLevel;
            ne_level[inx] = NE;
            correct[inx] = temp_cue;
            correct_lbl[inx] = (r >= cue_validity); # 0 if cue1, 1 if cue2
            acts_prob[inx,:] = actsProb; 
            
            # If the NE level goes above the ACh threshold (from Yu & Dayan)
            #   Network reset by setting ACH and NE levels back to initial values 
            if NE > AChLevel/(0.5 + AChLevel) and (not hasAChLesion) and (not hasNELesion):
                ACh = np.ones(numCues);
                NE = NE_RESET;
            
            inx += 1;
        
        #print(cue_validity);
        #print(AChLevel)        
    
    correct = correct[:inx];
    ach_level = ach_level[:inx,:];
    ne_level = ne_level[:inx];
    choice = choice[:inx];
    correct_lbl = correct_lbl[:inx];
    acts_prob = acts_prob[:inx,:];
    ach_avg_level = ach_avg_level[:inx];
    
    trend_choice = MNIST_choice_trend(choice,num_considered=num_considered,num_threshold=num_threshold)
    lag_trend,avg_lag = MNIST_trend_lag(trend_choice,major_cues,trial_intervals)
    
    prob_pred_wrong = times_pred_not_match_cue/inx;
    prob_pred_cue1_valid = times_match_cue1_valid/inx;
    prob_pred_cue2_valid = times_match_cue2_valid/inx;
    prob_pred_cue1_invalid = times_match_but_invalid/inx;
    prob_guess_not_match_cue = times_guess_not_match_cue/inx;
    prob_pred_not_match_guess = times_pred_not_match_guess/inx;
    prob_match_cue1_invalid_alter = times_match_cue1_invalid_alter/inx;
    
    all_prob = [validity_choices,trial_intervals,major_cues,lag_trend,avg_lag,\
                prob_pred_wrong,prob_pred_cue1_valid,prob_pred_cue2_valid,prob_pred_cue1_invalid,\
                prob_guess_not_match_cue,prob_pred_not_match_guess,prob_match_cue1_invalid_alter]
    
    #print('')
    print('{:d}/{:d} = {:.1f}% trials occurred when the predicted digit matched the cued digit but the cue was invalid'.format(times_match_but_invalid,inx,prob_pred_cue1_invalid*100));
    print('{:d}/{:d} = {:.1f}% trials occurred when the predicted digit did not match the cued digit.'.format(times_pred_not_match_cue,inx,prob_pred_wrong*100));
    print('    {:d}/{:d} trials occurred when the predicted digit did not match the guess-target digit.'.format(times_pred_not_match_guess,inx));
    print('    {:d}/{:d} trials occurred when the guess-target digit did not match the cued digit, i.e. when the guessed action was wrong.'.format(times_guess_not_match_cue,inx));
    print('{:d}/{:d} = {:.1f}% trials occurred when the predicted digit matches the major cued digit and is valid'.format(times_match_cue1_valid,inx,prob_pred_cue1_valid*100));
    print('{:d}/{:d} = {:.1f}% trials occurred when the predicted digit matches the minor cued digit and is valid'.format(times_match_cue2_valid,inx,prob_pred_cue2_valid*100));
    if alterValid:
        print('{:d}/{:d} = {:.1f}% trials occurred when the predicted digit matches the major cued digit but is invalidly altered'.format(times_match_cue1_invalid_alter,inx,prob_match_cue1_invalid_alter*100));
    print('Lag length for stable prediction of each major cue is: '+str(lag_trend)+', and mean of lag length (excluding consistently same cue) is '+str(avg_lag))
    
    return correct, choice, ach_level, ach_avg_level, ne_level, correct_lbl, acts_prob, all_prob



def plot_MNIST_uncertainty(correct, choice, ach_level, ach_avg_level, ne_level, correct_lbl, acts_prob, validity_choices=[0.99,0.85,0.70],moreSubplots=False):
    font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
    plt.rc('font', **font)
    
    ind_cue1, = np.where(correct_lbl == 0)
    ind_cue2, = np.where(correct_lbl == 1)
    if moreSubplots:
        f = plt.figure(figsize=[20,18])
        plt.subplot(5,1,1)
    else:
        f = plt.figure(figsize=[20,15])#plt.figure(figsize=[20,11])
        plt.subplot(3,1,1)
    plt.title('Correct Trials (validity: '+str(validity_choices)+')')
    plt.plot(np.arange(len(choice)),choice,'y*',markersize=8)
    plt.plot(ind_cue1,correct[ind_cue1],'r+',markersize=2)
    plt.plot(ind_cue2,correct[ind_cue2],'b+',markersize=2)
    plt.xlim(0,len(correct))
    plt.yticks(np.arange(4), ('even', 'odd', 'low', 'high'))
    plt.legend(('choice', 'major goal', 'minor goal'), loc='right')
    
    if moreSubplots:
        plt.subplot(5,1,2)
    else:
        plt.subplot(3,1,2)
    plt.title('Noradrenergic Level')
    plt.plot(ne_level)
    plt.xlim(0,len(ne_level))
    plt.ylim(0,1)
    
    if moreSubplots:
        plt.subplot(5,1,3)
        plt.title('Average Cholinergic Level')
        plt.plot(ach_avg_level)
        plt.xlim(0,len(ach_avg_level))
    
    if moreSubplots:
        plt.subplot(5,1,4)
    else:
        plt.subplot(3,1,3)
    plt.title('Cholinergic Level')
    plt.plot(ach_level)
    plt.legend(('even', 'odd', 'low', 'high'), loc='right')
    plt.xlim(0,ach_level.shape[0])
    
    if moreSubplots:
        plt.subplot(5,1,5)
        plt.title('Action Probability')
        plt.plot(acts_prob)
        plt.legend(('even', 'odd', 'low', 'high'), loc='right')
        plt.xlim(0,acts_prob.shape[0])
    
    plt.show()
       
    return None


def average_probs(all_case_probs):
    case_avg_lag_length = []
    cases_prob_pred_wrong = []
    cases_prob_pred_cue1_valid = []
    cases_prob_pred_cue2_valid =[] 
    cases_prob_pred_cue1_invalid = []
    cases_prob_guess_not_match_cue = []
    cases_prob_pred_not_match_guess = []
    cases_prob_match_cue1_invalid_alter = []
    for i in range(len(all_case_probs)):
        case_avg_lag_length.append(all_case_probs[i][4])
        cases_prob_pred_wrong.append(all_case_probs[i][5])
        cases_prob_pred_cue1_valid.append(all_case_probs[i][6])
        cases_prob_pred_cue2_valid.append(all_case_probs[i][7])
        cases_prob_pred_cue1_invalid.append(all_case_probs[i][8])
        cases_prob_guess_not_match_cue.append(all_case_probs[i][9])
        cases_prob_pred_not_match_guess.append(all_case_probs[i][10])
        cases_prob_match_cue1_invalid_alter.append(all_case_probs[i][11])
    averages=[]
    averages.append(np.mean(case_avg_lag_length))
    averages.append(np.mean(cases_prob_pred_wrong))
    averages.append(np.mean(cases_prob_pred_cue1_valid))
    averages.append(np.mean(cases_prob_pred_cue2_valid))
    averages.append(np.mean(cases_prob_pred_cue1_invalid))
    averages.append(np.mean(cases_prob_guess_not_match_cue))
    averages.append(np.mean(cases_prob_pred_not_match_guess))
    averages.append(np.mean(cases_prob_match_cue1_invalid_alter))
    print('Average lag length is: {:d}'.format(int(averages[0])))
    print('Average prob pred wrong, prob pred cue A valid, prob pred cue B valid, prob pred cue A invalid,')
    print('    prob guess not match cue, prob pred not match guess, prob match cue A invalid alter: ')
    print(averages[1:])
    
    return averages