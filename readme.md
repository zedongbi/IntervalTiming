# Understanding the computational of time using neural network models
By Zedong Bi and Changsong Zhou. arXiv: 1910.05546

## Dependencies
python 3.6, pytorch, sklearn, numba

## Contents
The folder 'core' contains most models.

The folder 'core_multitasking' contains the models used to study the effect
of timing transfer to the strength of temporal signals in non-timing tasks.

The folder 'core_feedback' contains the models used to study how feedback influences the 
strength of temporal signal in non-timing tasks.

The folder 'core_longinterval' contains the models used to study the temporal code when 
the to-be-produced intervals are longer than 1200 ms. 

## Reproducing the results of the paper
You can get the figures in the main text of the paper by running the files
 'fig2.py', 'fig3.py', 'fig4.py', 'fig5.py' and 'fig6.py' using python. 
 
## Pre-trained models
We provided pretrained models for analysis. You can download from https://drive.google.com/open?id=1vStFVYy3AL7eVD-pdoUIwDpjhXJ71E2E.

## Start to train
To train the models, simply run the files 'start_train.py',
'start_train_feedback.py', 'start_train_longinterval.py' and 'start_train_multitasking.py' 
using python. 

## Note
The convergence of training may be influenced by the 'initial_std' value of the hyperparameters, whose default
value is defined in 'default.py'. This value determines the standard deviation of the initialized values
of the non-diagonal recurrent weights, which, according to 'Orhan & Ma, Nat. Neurosci. (2019)', determines the 
 sequential activity after training. We found that large initial_std makes training hard to converge, with the cost function exploding
 at the first few training steps. By default,
 we set initial_std=0.3. However, we found the convergence of training may depend on the operating system or hardware environment.
 Therefore, if you find the training is hard to converge in your environment, reduce this number slightly. 

## Acknowledgement
This code is impossible without the following papers:

(1) G. R. Yang et al. Task representations in neural networks trained to
perform many cognitive tasks. Nat. Neurosci., 22, 297 (2019).

(2) A. E. Orhan and W. J. Ma. A diverse range of factors affect the nature
of neural representations underlying
short-term memory. Nat. Neurosci., 22, 275 (2019).


 
