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

Learning algorithm:
 1)***Perceptron Algorithm***      
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


2）***Pegasos Algorithm***  
if $y^{(i)} \cdot (\theta \cdot x^{(i)}) \leq 1$, then update $\theta = (1 - \eta \lambda) \theta + \eta y^{(i)} x^{(i)}$; else, update $\theta = (1 - \eta \lambda) \theta$


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
if loss > 0: $\theta \leftarrow \theta - \eta (- y^{(i)} \cdot x^{(i)} + \lambda \theta)$;  
if loss = 0, $\theta \leftarrow \theta - \eta \lambda \theta$.   
Differently from perceptron, $\theta$ is updated even when there is no mistake.    


***Support vector machine***: SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.


#------Lecture 4. Tuning the Regularization Hyperparameter by Cross Validation ------#  
***Supervised Learning***     
Objective function (J = Loss + regularization($\alpha$R)) --> hyperparameter ($\alpha$, not determined through the optimization of J) --> cross validation     

(1) Support Vector Machine --> maximize the margin (more generalization)    
$D: {(x_i, y_i)} \ i = 1, 2, ..., n$   
distant from point i to the decision boundary: $\gamma = \frac{y_i \cdot \ (\theta \cdot x_i + \theta_0)}{||\theta||}$    
margin, $d = \min_{x_i, y_i \in D} \ \gamma (x_i, y_i, \theta, \theta_0)$

***Objective function = hinge loss + regularization***
***hinge loss*** = $L_h = f(\frac{\gamma}{|gamma_{ref}}) = 1 - \frac{\gamma}{|gamma_{ref}}, \gamma \leq \gamma_{ref}; 0, \ o.w $  
<img src="https://user-images.githubusercontent.com/55688631/220841162-cadde400-63b0-4b96-9576-9e86fcf05bda.png" width="400" height="whatever">

***objective function***     
$J(\theta, \theta_0) = \frac{1}{n} \Sigma_{i=1}^n Loss_h (\frac{\gamma}{\gamma_{ref}}) + \alpha \frac{1}{||\gamma||^2}$   

--> Maximize the margin   
$\theta$ can be scaled by any constant w.o changing the decision boundary.
$\gamma_{ref} = \frac{y^M \cdot \ (\theta \cdot x^M + \theta_0)}{||\theta||}$    
scale $\theta$ st. $(y^M \cdot \ (\theta \cdot x^M + \theta_0) = 1$    
therefore, $\gamma_{ref} = \frac{1}{\theta_0}$   
$J(\theta, \theta_0) = \frac{1}{n} \Sigma_{i=1}^n Loss_h (y^{(i)} \cdot (\theta \cdot x^{(i)} + \theta_0)) + \alpha ||\theta||^2$   

(2) Cross validation --> get $\alpha$   
n: segments      
<img src="https://user-images.githubusercontent.com/55688631/220850369-5d896612-46ed-4361-9a7b-d4d4ac6451a8.png" width="400" height="whatever">   
accuracy score: $S(\alpha) = \frac{1}{n} \Sigma_{i=1}^n S(\alpha_i)$    
$\alpha^{\*} = argmin_{\alpha} S(\alpha)$ 


---------------------------------------------------------------------
## Unit 2. Nonlinear Classification, Linear regression, Collaborative Filtering
#------Lecture 5. Linear regression ------#   
(1) ***Objective***   
1) Empirical risk 
$R_n (\theta) = \frac{1}{n} \Sigma_{i=1}^n Loss(y^{(i)} - \theta \cdot x^{(i)})$   
- Squared error loss 
$Loss(z) = \frac{z^2}{2}$       
$R_n (\theta) = \frac{1}{n} \Sigma_{i=1}^n Loss(y^{(i)} - \theta \cdot x^{(i)}) = \frac{1}{n} \Sigma_{i=1}^n \cdot \frac{1}{2} (y^{(i)} - \theta \cdot x^{(i)})^2$   
- Hinge loss  
$Loss_h (z) = 0, if z \geq 1; 1-z, o.w$

(2) Learning Algorithm: gradient based approach











 
