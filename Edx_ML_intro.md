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

(2) ***Learning Algorithm***: gradient based approach   
$\nabla_{\theta} \ \frac{1}{2} (y^{(i)} - \theta \cdot x^{(i)})^2 = -(y^{(i)} - \theta \cdot x^{(i)}) \cdot x^{(i)}$  
initialize: $\theta = 0$     
learning step: $\theta = \theta + \eta (y^{(i)} - \theta \cdot x^{(i)}) \cdot x^{(i)}$   

SGD vs GD: Remark on SGD versus GD: Stochastic gradient descent (SGD) and gradient descent (GD) differ in their batch sizes. For "very" convex problems, GD can converge in fewer steps than SGD (with the same timesteps for fair comparison, and also taking moving averages of SGD's answer, for fair comparison). But for general nonlinear problems, we can't make a universal claim about which takes fewer steps to converge. For example, the randomness in SGD can help us escape critical points such as saddle points that would get GD stuck.    

(3) ***Closed form solution***        
$\nabla_{n}(\theta) = - \frac{1}{n} \Sigma_{i=1}^n (y^{(i)} - \theta \cdot x^{(i)}) \cdot x^{(i)} = 0$,      
if $x_1, ..., x_n \ in \ ℝ^d, n ≫ d, then \ \theta = A^{-1} b, where \ A = \frac{1}{n} \Sigma_{i=1}^n x^{(i)} (x^{(i)})^T, b = \frac{1}{n} \Sigma_{i=1}^n y^{(i)} x^{(i)}$  

(4) ***Regularization*** (Ridge and Lasso)        
***Ridge Regression***    
$J_{\lambda, \theta} (\theta) = R_n(\theta) + \frac{\lambda}{2} ||\theta||^2$    
learning step: $\theta = (1 - \eta \lambda) \theta + \eta (y^{(i)} - \theta \cdot x^{(i)}) \cdot x^{(i)}$      


#------Lecture 5. Non-linear classifier ------#     
(1) Higher order feature vectors         
map $x \in \ ℝ^d$ to $\phi(x) \in ℝ^p$     
e.g $\phi(x) = \[\phi_1(x), \phi_2(x)\]^T = \[x, x^2\]^2$   
$\phi(x) = \[x_1, x_2, x_1^2 + x_2^2\]^T$: hyperbolic paraboloid

(2) Non-linear classifier     
$h(x; \theta, \theta_0) = sign(\theta \cdot \phi(x) + \theta_0)$    

(3) Kernels (inner product)   
Computing the inner product of two feature vectors can be computationally cheap.     
$\phi(x) = \[x_1, x_2, x_1^2 + x_2^2 + \sqrt{2} x_1 x_2\]^T$    
$\phi(x') = \[{x'}_1, {x'}_2, {x'}_1^2 + {x'}_2^2 + \sqrt{2} {x'}_1 {x'}_2\]^T$     
$K(x, x') = \phi(x) \cdot \phi(x') = (x, x') + (x, x')^2$    

In general, $K(x, x') = \phi(x) \cdot \phi(x') = (1 + x x')^p, p = 1, 2,...$   

(4) Kernel Perceptron Algorithm    
From ***Perceptron algorithm***, we get $\theta = \Sigma_{j=1}^n \alpha_j y^{(j)} \phi(x^{(j)})$    
$initial \ \theta = 0$   
$run \ through \ i = 1, ..., n$   
$\ if \ y^{(i)} \theta \cdot \phi(x^{(i)}) \leq 0,$   
$\ \ \theta \leftarrow \theta + y^{(i)} \phi(x^{(i)})$    
Here, $\theta \cdot \phi(x^{(i)}) = \Sigma_{j=1}^n \alpha_j y^{(j)} \phi(x^{(j)}) \cdot \phi(x^{(i)}) = \Sigma_{j=i}^n \alpha_j y^{(j)} \cdot K(x^{(j)}, x^{(i)})$   

(5) Kernel composition rules     
-  $K(x, x') = 1$ is a kernel.
-  Let $f: ℝ^d \rightarrow ℝ$ and $K(x, x')$ is a kernel, then so it $\tilde{K}(x, x') = f(x) K(x, x') f(x')$.
-  If $K_1(x, x')$ and $K_2(x, x')$ are kernels,  $K_1(x, x') + K_2(x, x')$ is a kernel.
-  If $K_1(x, x')$ and $K_2(x, x')$ are kernels,  $K_1(x, x') \cdot K_2(x, x')$ is a kernel.


























 
