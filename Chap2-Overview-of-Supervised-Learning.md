<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script> -->

# **Overview of Supervised Learning**

The goal is to use the inputs (predictors, independent variables) to predict the values of the outputs (responses, dependent variables). This exercise is called supervised learning.

## **Variable Types and Terminology**

-  Inputs vary in measurement type: Some methods are defined most naturally for quantitative inputs, some most naturally for qualitative and some for both.

- Ordered categorical: there is an ordering between the values, but no metric notion 

- Dummy variables: $K$-level qualitative variable is represented by a vector of $K$ binary variables
or bits, only one of which is “on” at a time. (symmetric in the levels of the factor.)

- Terminology:
  - Input variable: $X$, components $X_j$.
  - Quantitative outputs: $Y$
  - Qualitative outputs: $G$
  - Observed values of $X$: lowercase, $x_i$ (a scalar or vector)
  - $N$ input $p$-vectors $x_i$: $\mathbf{X}\in \mathbb{R}^{N\times p}$
  - $\mathbf{x}_j$: all the observations on variable $X_j$ ($N$-vector).
  - All vectors are assumed to be column vectors, the $i$th row of $\mathbf{X}$ is $x_i^T$.

## **Least Squareds and Nearest Neighbors**

The linear model makes huge assumptions about structure and yields stable but possibly inaccurate predictions. The method of k-nearest neighbors makes very mild structural assumptions: its predictions are often accurate but can be unstable.

### **Linear Moders and Least Squares**

Given a vector of inputs $X^T=(1, X_1, X_2, ..., X_p)$, the output $Y$ is predicted via the model

$$\tag{2.1}
\hat{Y} = \hat{\beta_0} + \sum_{j=1}^p X_j\hat{\beta_j} = X^T\hat{\beta}.
$$

In general $\hat{Y}$ can be a $K$-vector, in which case $\beta$ would be a $p\times K$ matrix of coefficients. In the $(p+1)$-dimensional input-output space, $(X,\hat{Y})$ represents a hyperplane.

The most popular method for fiting the linear model to a set of training data is *least squares*, that is, the coefficients $\beta$ is picked to minimize the residual sum of squares
$$\tag{2.2}
\text{RSS}(\beta) = \sum_{i=1}^N(y_i-x_i^T\beta)^2=(\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta),
$$
where $\mathbf{X}$ is an $N\times p$ matrix with each row an input vector, and $\mathbf{y}$ is an $N$-vector of the outputs in the training set. Differentiating w.r.t $\beta$ we get the *normal equations*

$$\tag{2.3}
\mathbf{X}^T(\mathbf{y}-\mathbf{X}\beta) = 0.
$$

If $\mathbf{X}^T\mathbf{X}$ is nonsingular, then the unique solution is given by 
$$\tag{2.4}
\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y},
$$
and the fitted value at the $i$th input $x_i$ is $\hat{y}_i=x_i^T\hat{\beta}$.

Figure 2.1 shows a scatterplot of training data on a pair of inputs $X_1$ and $X_2$. The two predicted classeds are separated by the *decision boundary* $\{x:x^T\hat{\beta}=0.5\}$.
<div align=center>
<img src="pic/figure2.1.png" width="100%">
</div>

**Where did the constructed data come from?**

- bivariate Gaussian distributions with uncorrelated components and different means. (linear decision boundary is the best.)
- a mixture of 10 low-variance Gaussian distributions, with individual means themselves distributed as Gaussian. (nonlinear and disjoint is the best.)



### **Nearest-Neighbor Methods**

Using the observations in the training set $\mathcal{T}$ closest in input space to $x$ to form $\hat{Y}$. The $k$-nearest neighbor fit for $\hat{Y}$ is defined as follows:
$$
\hat{Y}(x)=\frac{1}{k} \sum_{x_i\in N_k(x)} y_i,
$$
where $N_k(x)$ is the neighborhood of $x$ defined by the $k$ closest points $x_i$ in the training sample. Closeness implies a metric, which for the momoent we assume is Euclidean distance.

<div align=center>
<img src="pic/figure2.2.png" width="100%">
</div>

In Figure 2.2, we see that far fewer training observations are misclassified than in Figure 2.1. A little thought suggests that the error on the training data should be approximately an increasing function of $k$, and will always be $0$ for $k=1$.

The effctive number of parameters for KNN is $N/k$ and is generally bigger than $p$, and decreases with increasing $k$. There would be $N/k$ neighborhoods and fit one parameter (a mean) in each nieghborhood.

If we use sum-of-squared errors on the training set as criterion for picking $k$, we would always pick $k=1$.

For the mixture Scenario 2, it seems that KNN would be more apropriate, while for Gaussian data the decision boundaries of k-nearest neighbors would be unnecessarily noisy.



### **From Least Squares to Nearest Neighbors**

- Linear model: decision boundary is very smooth, stable to fit, rely on the assumption that a liear decision boundary is appropriate. **Low variance and potentially high bias**. (scenario 1)
- KNN: do not rely on any stringent assumptions about the underlying data, can adapt to any situation, depends on a handful of input points and their particular positions, wiggly and unstable. **High variance and low bias**. (scenario 2)
- The variants of linear model and KNN: 
  
  - Kernel methods use weights that decrease smoothly to zero with dis- tance from the target point, rather than the effective 0/1 weights used by $k$-nearest neighbors.
  - In high-dimensional spaces the distance kernels are modified to em- phasize some variable more than others.
  - Local regression fits linear models by locally weighted least squares, rather than fitting constants locally.
  - Linear models fit to a basis expansion of the original inputs allow arbitrarily complex models.
  - Projection pursuit and neural network models consist of sums of non- linearly transformed linear models.

## **Statistical Decision Theory**

## **Local Methods in High Dimensions**

## 

