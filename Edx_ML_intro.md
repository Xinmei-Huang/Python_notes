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

#------L1. l2. linear classifiers ------#  
A linear classifier $h, h(x; \theta_0, \theta) = sign(\theta \cdot x + \theta_0)$, i.e. the sign of the dot product of $\theta$ and $x$ plus $\theta_0$.   
Linear separability: $y_i \cdot h(x_i) > 0$ for all $i$.  

(1) Algorithm:    
Training error: $\varepsilon_n(\theta_0, \theta) = \frac{1}{n} \ \Sigma_{i=1}^n \[\[ y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0) \leq 0 \]\]$  
where $\[\[ \cdot \]\]$ returns 1 if the logical expression in the argument is true, and zero otherwis  

Learning algorithm: Perceptron Algorithm   
initialize $\theta$ and $\theta_0$ with 0  
> $\theta$ = $\theta$ (vector)  
> $\theta_0$ = 0 (scalar)  
> totally T epoches to iterate  
>> for t = 1 .. T do                       
>> (totally m data points)         
>>> for i = 1 .. m do    
>>> (misclassify data points)                             
>>> if $y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0) \leq 0$        
>>> then  
>>> $\ \theta  = \theta + y^{(i)} \cdot x^{(i)}$   
>>> $\ \theta_0 = \theta_0 + y^{(i)}$   
>return $\theta, \theta_0$  \


#------Lecture 3. Hinge loss, Margin boundaries and Regularization ------#    
***Decision boundary*** is the set of points $x$ which satisfy: $\theta \cdot x + \theta_0 = 0$   
***Margin Boundary*** is the set of points $x$ which satisfy: $\theta \cdot x + \theta_0 = \pm 1$    
So, the distance from the decision boundary to the margin boundary is $\frac{1}{||\theta||}$.   

***Regularization***:  $max \lbrace\frac{1}{||\theta||}\rbrace$ = $min \lbrace\frac{1}{2}||\theta||\rbrace$      
***Hinge loss***: $Loss_h (y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0)) = 0, (if \ z \geq 1); 1-z, (if \ z < 1)$, where $z = y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0)$     
***Objectives = average loss + regularization***        
$J(\theta, \theta_0) = \frac{1}{n} \Sigma_{i=1}^n Loss_h (y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0)) + \frac{\lambda}{2}||\theta||^2$,
where average loss: $\frac{1}{n} \Sigma_{i=1}^n Loss_h (y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0))$, regularization term: $\frac{\lambda}{2}||\theta||^2$, regularization paramter: $\lambda > 0 $.   


#------Lecture 4. Linear Classification and Generalization ------#      
Split training set into training set + validation set, and assess the loss of validation set.   

***Gradient descent***: Start $\theta$ at an arbitrary location: $\theta \leftarrow \theta_{start}$,    
update $\theta$ repeatedly with $\theta \leftarrow \theta - \eta \frac{\partial J(\theta, \theta_0)}{\partial \theta}$ until $\theta$  does not change significantly.  

***Stochastic gradient descent***: (looking at each individual item randomly) With stochastic gradient descent, we choose $i \in {1, ..., n}$ at random and update $\theta$ such that     
$\theta \leftarrow \theta - \eta \nabla \[Loss_h (y^{(i)} \\cdot (\theta \cdot x^{(i)} + \theta_0)) + \frac{\lambda}{2}||\theta||^2\]$












 
