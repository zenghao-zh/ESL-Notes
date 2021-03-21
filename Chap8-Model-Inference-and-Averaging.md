<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script> -->

# **Model Inference and Averaging**

## **8.1 Introduction**

For most of this book, the fitting (learning) of models has been achieved byminimizing a sum of squares for regression, or by minimizing cross-entropy for classification. In fact, both of these minimizations are instances of the maximum likelihood approach to fitting. In this chapter we provide a general exposition of the maximum likelihood approach, as well as the Bayesian method for inference.

## **8.2 The Bootstrap and Maximum Likelihood Methods**

### **Bootstrap**
- Nonparametric bootstrap: draw $B$ datasets each of size $N$ with replacement from training data. To each bootstrap dataset $\mathbf{Z}^*$, find $\hat{y}^*$. It uses the raw data, not a specific parametric model, to generate new datasets.
- Parameteric bootstrap: suppose that the model has additive Gaussian errors. One can add a noise to the predicted values $y^*_i = \hat{y}_i + \varepsilon_i^*$. This process is repeated $B$ times. The resulting bootstrap datasets have the form $(x_1,y_1^*), ..., (x_N, y_N^*)$, from this, we can obtain $\hat{y}^*$. 
 
In least square case, a function estimated from a bootstrap sample $\mathbf{y}^*$ has distribution
  $$
    \hat{y}^*(x) \sim N(\hat{y}(x), h(x)^T(H^TH)^{-1}h(x)\hat{\sigma}^2),
  $$


### **Maximum Likelihood Inference**
It turns out that the parametric bootstrap agrees with least squares in the previous example because the model has additive Gaussian errors. In general, the parametric bootstrap agrees not with least squares but with maximum likelihood.

The probability density or probability mass function with parameter $\theta$ for our observations
$$\tag{8.8}
z_i\sim g_{\theta}(z).
$$

Maximum lilkelihod is based on the *likelihood fucntion*, given by
$$\tag{8.11}
L(\theta; \mathbf{Z}) = \prod_{i=1}^Ng_{\theta}(z_i),
$$

Denote the logarithm of $L(\theta; \mathbf{Z})$ by
$$\tag{8.12}
\ell(\theta;\mathbf{Z}) = \sum_{i=1}^N\ell(\theta;z_i) = \sum_{i=1}^N\log g_{\theta}(z_i).
$$

Assuming that the likelihood takes its maximum in the interior of the parameter space. The score function is defined by 
$$\tag{8.13}
\frac{\partial \ell(\hat{\theta}; \mathbf{Z})}{\partial \theta} = 0.
$$

The *information matrix* is 

$$\tag{8.14}
\mathbf{I}(\theta) = -\sum_{i=1}^N\frac{\partial^2 \ell(\theta;z_i)}{\partial \theta \partial \theta^T}.
$$
When $\mathbf{I}(\theta)$ is evaluated at $\theta = \hat{\theta}$, it is often called the *observed information*. The *Fisher information* is 
$$\tag{8.15}
  \mathbf{i}(\theta) = \text{E}_{\theta}[\mathbf{I}(\theta)].
$$
Finally, let $\theta_{0}$ denote the true value of $\theta$. A standard result says that the sampling distribution of the maximum likelihood estimator has a limiting normal distribution
$$\tag{8.16}
\hat{\theta} \to N(\theta_0, \mathbf{i}(\theta_0)^{-1}),
$$
as $N\to \infty$. Here we are independently sampling from $g_{\theta_0}(z)$. This suggests that the sampling distribution of $\hat{\theta}$ (maximum likelihood estimate from the observed data) may be approximated by
$$\tag{8.17}
N(\hat{\theta}, \mathbf{i}(\hat{\theta})^{-1}) \text{ or } N(\hat{\theta}, \mathbf{I}(\hat{\theta})^{-1}).
$$

### **Bootstrap versus Maximum Likelihood**
- In essence the bootstrap is a computer implementation of nonparametric or parametric maximum likelihood. 

## **8.3 Bayesian Methods**

Specifying a sampling model $\Pr(\mathbf{Z}|\theta)$ for data given the parameters and a prior distribution for the parameters $\Pr(\theta)$ reflecting our knowledge about $\theta$. The posterior distribution 
$$\tag{8.23}
\Pr(\theta|\mathbf{Z}) = \frac{\Pr(\mathbf{Z}|\theta)\cdot \Pr(\theta)}{\int \Pr(\mathbf{Z}|\theta)\cdot \Pr(\theta)d\theta}.
$$
The posterior distribnution also provides the basis for predicting the values of a future observation $z^{\text{new}}$, via the predictive distribution
$$\tag{8.24}
\Pr(z^{\text{new}}|\mathbf{Z}) = \int \Pr(z^{\text{new}}|\theta)\cdot \Pr(\theta|\mathbf{Z}) d\theta.
$$

B-spline example: For Gaussian distribution prior with the prior correltation matrix $\Sigma$ and variance $\tau$, $\beta\sim N(0,\tau\Sigma)$,   the posterior distribution for $\beta$ is also Gaussian. As $\tau \to \infty$, the
posterior distribution and the bootstrap distribution coincide. The distribution with $\tau\to\infty$ is called a noninformative prior for $\theta$. 

 In Gaussian models, maximum likelihood and parametric bootstrap analyses tend to agree with Bayesian analyses that use a noninformative prior for the free parameters. These tend to agree, because with a constant prior, the posterior distribution is proportional to the likelihood. This correspondence also extends to the nonparametric case, where the nonparametric bootstrap approximates a noninformative Bayes analysis; Section 8.4 has the details.

## **8.4 Relationship Between the Bootstrap and Bayesian Inference**

Let $z\sim N(\theta, 1)$, given the prior $\theta\sim N(0,\tau)$, the posterior distribution with maximum likelihood estimate $\hat{\theta}=z$
$$\tag{8.30}
\theta|z \sim N\bigg(\frac{z}{1+1/\tau}, \frac{1}{1+1/\tau}\bigg).
$$
When $\tau\to \infty$, $\theta|z\sim N(z,1)$, this is the same as a parameteric bootstrap distribution in which we generate bootstrap values $z^∗$ from the maximum likelihood estimate of the sampling density $N(z,1)$.

There are three ingredients that make this correspondence work:

1. The choice of noninformative prior for $\theta$.
2. The dependence of the log-likelihood $\ell(\theta;\mathbf{Z})$ on data $\mathbf{Z}$ only through the maximum likelihood estimate $\hat{\theta}$. Hence we can write the log-likelihood as $\ell(\theta;\hat{\theta})$. 
3. The symmetry of the log-likelihood in $\theta$ and $\hat{\theta}$, that is $\ell(\theta;\hat{\theta}) = \ell(\hat{\theta};\theta)+\text{constant}$.

Properties (2) and (3) essentially only hold for the Gaussian distribution. However, they also hold approximately for the multinomial distribution, leading to a correspondence between the nonparametric bootstrap and Bayes inference. 

In this sense, the bootstrap distribution represents an (approximate) nonparametric, noninformative posterior distribution for our parameter. But this bootstrap distribution is obtained painlessly—without having to formally specify a prior and without having to sample from the posterior distribution. Hence we might think of the bootstrap distribution as a “poor man’s” Bayes posterior. By perturbing the data, the bootstrap approximates the Bayesian effect of perturbing the parameters, and is typically much simpler to carry out.

## EM (......)

### **8.7 Bagging**

Earlier we introduced the bootstrap as a way of assessing the accuracy of a parameter estimate or a prediction. Here we show how to use the bootstrap to improve the estimate or prediction itself. In Section 8.4 we investigated the relationship between the bootstrap and Bayes approaches, and found that the bootstrap mean is approximately a posterior average.

Consider first the regression problem. Suppose we fit a model to our training data $\mathbf{Z} = \{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$, obtaining the prediction $\hat{f}(x)$ at input $x$. Bootstrap aggregation or **bagging** averages this prediction over a collection of bootstrap samples, thereby reducing its variance. For each bootstrap sample $\mathbf{Z}^{*b}, b =1,2,...,B$, we fit our model, giving prediciton $\hat{f}^{*b}(x)$. The bagging estimate is defined by
$$\tag{8.51}
\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^B\hat{f}^{*b}(x).
$$
Denote by $\hat{\mathcal{P}}$ the empirical distribution putting equal probability $1/N$ on
each of the data points $(x_i , y_i)$. In fact the "true" bagging estimate is defined by $\text{E}_{\mathcal{\hat{P}}}\hat{f}^{*}(x)$, where $\mathbf{Z}^*=\{(x_1^*,y_1^*),(x_2^*,y_2^*),...,(x_N^*,y_N^*)\}$ and each $(x_i^*,y_i^*)\sim \hat{\mathcal{P}}$. The bagged estimate (8.51) will differ from the original estimate $\hat{f}(x)$
only when the latter is a nonlinear or adaptive function of the data.
> Exercise 8.4

Often we require the class-probability estimates at $x$, rather than the classifications themselves. It is tempting to treat the voting proportions $p_k(x)$ as estimates of these probabilities.