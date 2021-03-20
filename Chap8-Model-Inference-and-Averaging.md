<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script> -->

# **Model Inference and Averaging**

## **8.1 Introduction**

For most of this book, the fitting (learning) of models has been achieved byminimizing a sum of squares for regression, or by minimizing cross-entropy for classification. In fact, both of these minimizations are instances of the maximum likelihood approach to fitting. In this chapter we provide a general exposition of the maximum likelihood approach, as well as the Bayesian method for inference.

## **8.2 The Bootstrap and Maximum Likelihood Methods**

### Bootstrap
- Nonparametric bootstrap: draw $B$ datasets each of size $N$ with replacement from training data. To each bootstrap dataset $\mathbf{Z}^*$, find $\hat{y}^*$. It uses the raw data, not a specific parametric model, to generate new datasets.
- Parameteric bootstrap: suppose that the model has additive Gaussian errors. One can add a noise to the predicted values $y^*_i = \hat{y}_i + \varepsilon_i^*$. This process is repeated $B$ times. The resulting bootstrap datasets have the form $(x_1,y_1^*), ..., (x_N, y_N^*)$, from this, we can obtain $\hat{y}^*$. 
 
In least square case, a function estimated from a bootstrap has distribution
  $$
    \hat{y}^*(x) \sim N(\hat{y}(x), h(x)^T(H^TH)^{-1}h(x)\hat{\sigma}^2),
  $$


### Maximum Likelihood Inference
It turns out that the parametric bootstrap agrees with least squares in the previous example because the model has additive Gaussian errors. In general, the parametric bootstrap agrees not with least squares but with maximum likelihood.