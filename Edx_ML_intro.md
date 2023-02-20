# Edx_ML_intro

---------------------------------------------------------------------
## Unit 0. Python set up

conda updates:  
https://github.com/ContinuumIO/anaconda-issues/issues/12135 [bluesblaz]  
https://stackoverflow.com/questions/45197777/how-do-i-update-anaconda  

Install packages:  
1. anaconda prompt:   
-activate (environment)  
-conda install ...  
2. Anaconda   

Pycharm
1. add configuration: add new-- choose python-- find the path to the xxx.py file  
2. add interpreter: excutable conda: select "conda.exe", chech the path by "which conda" in anaconda prompt.  
3. check the interpreter in the configuration  

---------------------------------------------------------------------
## Unit 1. Linear classifier

#------L1. linear classifiers ------#  
A linear classifier $h, h(x; \theta_0, \theta) = sign(\theta \cdot x + \theta_0)$, i.e. the sign of the dot product of $\theta$ and $x$ plus $\theta_0$.   
Linear separability: $y_i \cdot h(x_i) > 0$ for all $i$.  

- Algorithem:    
Training error: $\varepsilon_n(\theta_0, \theta) = \frac{1}{n} \ \Sigma_{i=1}^n \[\[ y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0) \leq 0 \]\]$  
where $\[\[ \cdot \]\]$returns 1 if the logical expression in the argument is true, and zero otherwis  
