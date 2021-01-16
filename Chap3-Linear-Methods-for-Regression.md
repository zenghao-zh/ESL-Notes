<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script> -->

# **Linear Methods for Regression**

## Linear regression model
- Assuming that the regression function $E(Y|X)$ is linear in the inputs $X_1,...,X_p$.
- Simple and provide an adequate and interpretable description of how the inputs affect the output.
- Outperform in small numbers training cases, low signal-to-noise ratio or sparse data.
- In Chapter 5, basis-funciton methods will be discussed.

## Least squares

Suppose that the regression function $E(Y|X)$ is linear, given input vector $X^T = (X_1, X_2, ..., X_p)$, real-valued output $Y$, the linear regression model has the form

$$\tag{3.1}
f(X) = \beta_0+ \sum_{j=1}^p X_j\beta_j.
$$

The variables $X_j$ can come from different sources:
- quantitative inputs and its transformations (log, square-root or square);
- basis expansions, (e.g. $X_2=X_1^2, X_3 = X_1^3$, a polynomial representation);
- numeric or "dummy" coding of the levels of qualitative inputs.
- interactions between variables (e.g. $X_3=X_1X_2$)

Suppose we have a set of training data $(x_1, y_1), ..., (x_N, y_N)$ to estimate the parameters $\beta$, where $x_i= (x_{i1}, x_{i2}, ..., x_{ip})^T$, the least squares minimize the residual sum of squares (RSS)
$$\tag{3.2}
\begin{aligned}
\text{RSS}(\beta) &= \sum_{i=1}^N(y_i-f(x_i))^2 \\
&= \sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2 \\
&= (\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta).
\end{aligned}
$$

**Statistical point of view**

- this criterion is reasonable if $(x_i, y_i)$ independent random draws from their population;
- or if the $y_i$'s are conditionally independent given the inputs $x_i$;
- the criterion measures the average lack of fit.

**The columns of $\mathbf{X}$ are linearly independent**
If $\mathbf{X}$ has full column rank, and hence $\mathbf{X}^T\mathbf{X}$ is positive definite, we can obtain the unique solution
$$\tag{3.3}
\hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}.
$$

The predicted values at an input vector $x_0$ are given by $\hat{f}(x_0)=(1:x_0)^T\hat{\beta}$; the fitted values at the training inputs are
$$\tag{3.4}
\hat{\mathbf{y}} = \mathbf{X}\hat{\beta} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}, 
$$
where $\hat{y}_i = \hat{f}(x_i)$. The matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is sometimes called the "hat" matrix. The residual vector $\mathbf{y}-\hat{\mathbf{y}}$ is orthogonal to this subspace.

<div align=center>
<img src="pic/figure3.2.png" width="61.8%">
</div>

**The columns of $\mathbf{X}$ are not linearly independent**

(e.g. $x_2=3x_1$)

- $\mathbf{X}^T\mathbf{X}$ is singular and $\hat{\beta}$ are not uniquely defined.
- $\hat{\mathbf{y}}=\mathbf{X}\hat{\beta}$ are still the projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$ (just more than one way to express). 
- The non-full-rank case occurs most often when one or more qualitative inputs are coded in a redundant fashion.
- There is usually a natural way to resolve the non-unique representation, by recoding and/or dropping redundant columns in $\mathbf{X}$. (most regression software packages detect and removing them automatically)
- In image or signal analysis, sometimes $p > N$ (rank deficiencies), the features can be reduced by filtering or regularization (Section 5.2.3 and Chapter 18.)

**The sampling properties of $\hat{\beta}$**

We now assume that the observations $y_i$ are uncorrelated and have constant variance $\sigma^2$, and that the $x_i$ are fixed (non random). The variance-covariance matrix of the least squares parameter estimates is 

$$\tag{3.5}
Var(\hat{\beta}) = (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2.
$$

Typically one estimates the variance $\sigma^2$ unbiased by
$$\tag{3.6}
\hat{\sigma}^2 = \frac{1}{N-p-1}\sum_{i=1}^N(y_i-\hat{y}_i)^2.
$$
(3.6) can be shown with the following assumption (In fact, the noise could not be Gaussian).

We now assume that (3.1) is the correct model, and the deviations of $Y$ around its expectation are additive and Gaussian,hence 
$$\tag{3.7}
Y= \beta_0+\sum_{j=1}^pX_j\beta_j +\varepsilon, 
$$
where the error $\varepsilon\sim N(0,\sigma^2)$. It is easy to show that 

$$\tag{3.8}
\hat{\beta}\sim N(\beta, (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2).
$$

This is a multivariate normal distribution with mean vector and variance–covariance matrix as shown. Also

$$\tag{3.9}
(N-p-1)\hat{\sigma}^2 \sim \sigma^2 \mathcal{X}^2_{N-p-1}.
$$
In addition $\hat{\beta}$ and $\hat{\sigma}^2$ are statistically independent.

> *proof of (3.6).*   
>  First note that 
> $$\sum_{i=1}^N(y_i-\hat{y}_i)^2 =\|\mathbf{y}-\mathbf{X}\hat{\beta}\|^2.$$
> We have
> $$
\begin{aligned}
\mathbf{y}-\mathbf{X}\hat{\beta} &= \mathbf{X}\beta+\varepsilon- \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{X}\beta+\varepsilon)\\
&=(I_N-H)\varepsilon 
\end{aligned} $$ 
> Hence, 
> $$ E(\|\mathbf{y}-\mathbf{X}\hat{\beta}\|^2) = E(\varepsilon^T(I_N-H)^T(I_N-H)\varepsilon)= E(\varepsilon^T(I_N-H)\varepsilon) $$
> Note that 
> $$
\varepsilon^T(I_N-H)\varepsilon = \sum_{i,j}\varepsilon_i\varepsilon_j(\delta_{ij}-H_{ij}) $$
>Thus, 
> $$
\begin{aligned}
E(\|\mathbf{y}-\mathbf{X}\hat{\beta}\|^2)&=
E(\varepsilon^T(I_N-H)\varepsilon)\\
&=\sum_{i=1}^N\sigma^2(\delta_{ii}-H_{ii})\\ 
&= \sigma^2(N-\text{trace}(H))\\
&= \sigma^2(N-p-1)
\end{aligned} $$

> *proof of (3.8).*  
> Since $\varepsilon\sim N(0, \sigma^2)$, then $\hat{\beta}$ is normal distribution,
> $$\begin{aligned}
E(\hat{\beta}) &= E((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{y})\\
&= E((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T (\mathbf{X}\beta+\varepsilon))\\
&=\beta\end{aligned}$$ 
> and
> $$
Var(\hat{\beta}) = Var((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}) = (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2 
$$

> *proof of (3.9).*  
> Since $\varepsilon\sim N(0, \sigma^2)$, from the proof of (3.6), we have
> $$
\mathbf{y}-\hat{\mathbf{y}}=\mathbf{y}-\mathbf{X}\hat{\beta} = (I_N-H)\varepsilon 
>$$
> Then, each element is normal distribution. According to the linearity of expectation, we obtain
> $$
E(\mathbf{y}-\hat{\mathbf{y}})=E(\mathbf{y}-\mathbf{X}\hat{\beta})= 0.
> $$
> From the proof of (3.6), we also have
> $$
E(\|\mathbf{y}-\hat{\mathbf{y}}\|^2)= E(\|\mathbf{y}-\mathbf{X}\hat{\beta}\|^2) = \sigma^2(N-p-1)
>$$
> Thus,  $(N-p-1)\hat{\sigma}^2\sim \sigma^2 \mathcal{X}^2_{N-p-1}$.

In addition $\hat{\beta}$ and $\hat{\sigma}^2$ are independent (In normal distribution, sample mean and sample variance are independent, refer it to prove that). To test the hypothesis that a particular coefficient $\beta_j=0$, we form the standardized coefficient or $Z$-score
$$\tag{3.10}
z_j = \frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{v_j}}, \quad v_j \text{ is the $j$th diagonal element of $(\mathbf{X}^T\mathbf{X})^{-1}$.}
$$

- $z_j\sim t_{N-p-1}$
- a large (absolute) value of $z_j$ will lead to rejection of this null hypothesis.
- If $\hat{\sigma}$ is replaced by a known value $\sigma$, then $z_j\sim N(0,1)$. The tail quantitles of $t$-distribution and a standard normal become negligible as the sample size increases, so we can use the normal quantiles to approximate it.
- we can isolate $\beta_j$ to obtain a $1-2\alpha$ confidence interval for $\beta_j$:
  $$\tag{3.11}
    (\hat{\beta}_j-z^{(1-\alpha)}v_j^{\frac{1}{2}}\hat{\sigma}, \hat{\beta}_j+z^{(1-\alpha)}v_j^{\frac{1}{2}}\hat{\sigma}), \quad \text{$z^{(1-\alpha)}$ is the $1-\alpha$ percentile of the normal distribution.}
  $$

When we need to test for the significance of groups of coefficients simultaneously. We can use the $F$ statistic, let $\text{RSS}_1$ is the residual sum of squares for the least squares fit of the bigger model with $p_1+1$ parameters, and $\text{RSS}_0$ the same for the nested smaller model with $p_0+1$ parameters, having $p_1-p_0$ parameters constrained to be zero.
$$\tag{3.12}
F= \frac{(\text{RSS}_0-\text{RSS}_1)/(p_1-p_0)}{\text{RSS}_1/(N-p_1-1)},
$$
The $F$ statistic measures the change in residual sum-of-squares per additional parameter in the bigger model, and it is normalized by an estimate of $\sigma^2$. Under the Gaussian assumptions, the null hypothesis that the smaller model is correct, the $F$ statistic will have a $F_{p_1-p_0,N-p_1-1}$ distribution. For large $N$, the quantiles of $F_{p_1-p_0, N-p_1-1}$ approach those of $\mathcal{X}^2_{p_1-p_0}/(p_1-p_0)$. 

(Exercise 3.1) $z_j$ are equivalent to the $F$ statistic for dropping the single coefficient $\beta_j$ from the model.

> *Proof of Exercise 3.1.*

We can obtain an approximate confidence set for the entire parameter vector $\beta$, namely
$$\tag{3.13}
C_{\ beta} = \{\beta| (\hat{\beta}-\beta)^T\mathbf{X}^T\mathbf{X}(\hat{\beta}-\beta) \leq \hat{\sigma}^2\mathcal{X}_{p+1}^2 \,^{(1-\alpha)}\}.
$$
This confidence set for $\beta$ generates a corresponding confidence set for the true function $f (x) = x^T\beta$, namely $\{x^T\beta|\beta \in C_{\beta} \}$ (Exercise 3.2).

>*proof of (3.13).*
> From (3.8), it is easy to show that
> $$
(\mathbf{X}^T\mathbf{X})^{1/2}(\hat{\beta}-\beta) \sim N(0, \sigma^2I_N). $$
> Then 
> $$
(\hat{\beta}-\beta)^T\mathbf{X}^T\mathbf{X}(\hat{\beta}-\beta) \sim \sigma^2\mathcal{X}^2_{p+1}. $$
> Combining this with (3.9), it gives
> $$
\frac{(\hat{\beta}-\beta)^T\mathbf{X}^T\mathbf{X}(\hat{\beta}-\beta)}{(p+1)\hat{\sigma}^2}\sim F_{p+1, N-p-1}. $$
> On the other hand, one can prove if $S\sim F_{m,n}, T =\lim_{n\to\infty} mS \sim \mathcal{X}^2_m$ by directly computing the limit of $mS$'s pdf, with the help of relation between gamma function and beta function and Stirling's formula. With this claim, we have
> $$
\frac{(\hat{\beta}-\beta)^T\mathbf{X}^T\mathbf{X}(\hat{\beta}-\beta)}{\hat{\sigma}^2}\sim \mathcal{X}^2_{p+1}\quad (N\to \infty).
$$

*Example: Prostate Cancer (Page 50).*

## The Gauss-Markov Theorem

**The least squares estimates of the parameters $\beta$ have the smallest variance among all linear unbiased estimates.**

We focus on estimation of any linear combination of the parameters $\theta= a^T\beta$. The least squares estimte of $a^T\beta$ is 
$$\tag{3.14}
\hat{\theta} = a^T\hat{\beta} = a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}.
$$

Considering $\mathbf{X}$ to be fixed, this is a linear function $\mathbf{c}_0^T\mathbf{y}$ of the response vector $\mathbf{y}$. If we assume that the linear model is correct, $a^T\hat{\beta}$ is unbiased since
$$\tag{3.15}
\begin{aligned}
E(a^T\hat{\beta}) &= E(a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y})\\
& = a^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}\beta\\
& = a^T\beta.
\end{aligned}
$$

(Exercise 3.3) The Gauss-Markov theorem states that if we have any other linear estimator $\tilde{\theta} = \mathbf{c}^T\mathbf{y}$ that is unbiased for $a^T\beta$, then 
$$\tag{3.16}
Var(a^T\hat{\beta}) \leq Var(\mathbf{c}^T\mathbf{y}).
$$
For simplicity we have stated the result in terms of estimation of a single parameter $a^T\beta$, but with a few more definitions one can state it in terms of the entire parameter vector $\beta$ (Exercise 3.3).

> *Proof of Exercise 3.3*

Consider the mean squared error of an estimator $\tilde{\theta}$ in estimating $\theta$:
$$\tag{3.17}
\text{MSE}(\tilde{\theta}) = E(\tilde{\theta}-\theta)^2 = Var(\tilde{\theta}) + (E(\tilde{\theta})-\theta)^2.
$$
There may well exist a biased estimator with smaller mean squared error, such an estimator would trade a little bias for a larger reduction in variance. Any method that shrinks or sets to zero some of the least squares coefficients may result in a biased estimate.

Mean squared error is intimately related to prediction accuracy, as discussed in Chapter 2. Consider the prediction of the new response at input $x_0$, $Y_0=f(x_0)+\varepsilon_0$. Then the expected prediciton error (EPE) of an estimate $\tilde{f}(x_0)=x_0^T\tilde{\beta}$ is 
$$\tag{3.18}
E(Y_0-\tilde{f}(x_0))^2 = E(x_0^T\tilde{\beta}-f(x_0))^2 + \sigma^2 = \text{MSE}(\tilde{f}(x_0))+ \sigma^2.
$$
Therefore, expected prediction error and mean squared error differ only by
the constant $\sigma^2$.

## Multiple Regression from Simple Univariate Regression

The linear model with $p>1$ inputs is called the *multiple linear regression model*. $p=1$ called *univariate* linear model. 

In convenient vector notation, let $\mathbf{y} = (y_1,...,y_N)^T, \mathbf{x} = (x_1,...,x_N)^T$, Then the univariate linear model with no intercepte ($Y=X\beta+\varepsilon$) have least squares estimate and residuals:
$$\tag{3.19}
\begin{aligned}
\hat{\beta}  &= \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\langle \mathbf{x}, \mathbf{x} \rangle}, \\
\mathbf{r}  &=  \mathbf{y} - \mathbf{x} \hat{\beta}.
\end{aligned}
$$

Suppoose that the inputs $\mathbf{x}_1,\mathbf{x}_2, ..., \mathbf{x}_p$ (the columns of the data matrix $\mathbf{X}$) are orthogonal; Then the mutiple least squares estimates $\hat{\beta}_j$ are equal to $\langle \mathbf{x}_j, \mathbf{y} \rangle / \langle \mathbf{x}_j, \mathbf{x}_j \rangle$.

Orthogonal inputs occur most often with balanced, designed experiments(where orthogonality is enforced), but almost never with observationaldata. Hence we will have to orthogonalize them in order to carry this idea further.

**Gram-Schmidt Procedure**

---
**Algorithm 3.1** *Regression by Successive Orthogonalization.*

---
1. Initialize $\mathbf{z}_0=\mathbf{x}_0=\mathbf{1}$.
2. For $j=1,2,...,p$
   
   Regress $\mathbf{x}_j$ on $\mathbf{z}_0, \mathbf{z}_1, ..., \mathbf{z}_{j-1}$ to produce coefficients $\hat{\gamma}_{\ell j}= \langle \mathbf{z}_{\ell}, \mathbf{x}_j\rangle / \langle \mathbf{z}_{\ell}, \mathbf{z}_{\ell} \rangle,\ell = 0,...,j-1$ and residual vector $\mathbf{z}_j = \mathbf{x}_j-\sum_{k=0}^{j-1}\hat{\gamma}_{kj}\mathbf{z}_k$.
3. Regress $\mathbf{y}$ on the residual $\mathbf{z}_p$ to give the estimate $\hat{\beta}_p$.

---

<div align=center>
<img src="pic/figure3.4.png" width="61.8%">
</div>

- The result of this algorithm is 
$$\tag{3.20}
\hat{\beta}_p=\frac{\langle \mathbf{z}_p, \mathbf{y} \rangle}{\langle \mathbf{z}_p, \mathbf{z}_p \rangle}.
$$
- The inputs $\mathbf{z}_0, \mathbf{z}_1,..., \mathbf{z}_{j-1}$ in step 2 are orthogonal.
- Since $\mathbf{z}_p$ alone involves $\mathbf{x}_p$ (with coefficient 1), the coefficient (3.20) is indeed the coefficient of $\mathbf{y}$ on $\mathbf{x}_p$.
- By rearranging the $\mathbf{x}_j$, any one of them could be in the last position, and a similar results holds.
- (Exercise 3.4) show that how the vector of least squares coefficients can be obtained from a single pass of the Gram–Schmidt procedure(Algorithm 3.1).
  > *Proof of Exercise 3.4.*
- From (3.20) we also obtain an alternate formula for the variance estimates (3.5), 
  $$\tag{3.21}
    Var(\hat{\beta}_p) = \frac{\sigma^2}{\langle \mathbf{z}_p, \mathbf{z}_p\rangle} = \frac{\sigma^2}{\|\mathbf{z}_p\|^2}.
  $$ 

   The precision with wich we can estimate $\hat{\beta}_p$ depends on the length of the residual vector $\mathbf{z}_p$; this represents how mush of $\mathbf{x}_p$ is unexplained by the other $\mathbf{x}_k$'s.


## Multiple outputs

Suppose we have multiple ouputs $Y_1, Y_2, ..., Y_K$ that we wish to predict from our inputs $X_0, X_1, X_2, ..., X_p$. We assume a linear model for each output
$$\tag{3.22}
Y_k  = f_k(X)+\varepsilon_k = \beta_{0k} + \sum_{j=1}^pX_j\beta_{jk}+\varepsilon_k
$$

With $N$ training cases we can writhe the model in matrix notation
$$\tag{3.23}
\mathbf{Y}  = \mathbf{X}\mathbf{B} + \mathbf{E}.
$$
Here $\mathbf{Y}$ is the $N\times K$ response matrix, with $ik$ entry $y_{ik}$, $\mathbf{X}$ is the $N\times (p+1)$ input matrix, $\mathbf{B}$ is the $(p+1) \times K$ matrix of parameters and $\mathbf{E}$ is the $N\times K$ matrix of errors. The residual sum of squares
$$\tag{3.24}
\text{RSS}(\mathbf{B}) = \sum_{k=1}^K\sum_{i=1}^N(y_{ik}-f_k(x_i))^2 = \text{trace}((\mathbf{Y}-\mathbf{X}\mathbf{B})^T(\mathbf{Y}-\mathbf{X}\mathbf{B})).
$$
The least squares estimates have exactly the same form as before
$$\tag{3.25}
\mathbf{\hat{B}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}.
$$
Hence the coefficients for the $k$th outcome are just the least squares estimates in the regression of $\mathbf{y}_k$ on $\mathbf{x}_0, \mathbf{x}_1,...,\mathbf{x}_p$. Multiple outputs do not affect one another’s least squares estimates.

If the errors $\varepsilon = (\varepsilon_1,...,\varepsilon_K )$ in (3.22) are correlated, then it might seem appropriate to modify (3.24) in favor of a multivariate version. Specifically, suppose $Cov(\varepsilon) = \mathbf{\Sigma}$, then the multivariate weighted criterion

$$\tag{3.25}
\text{RSS}(\mathbf{B}) = \sum_{i=1}^N(y_{i}-f(x_i))^T\mathbf{\Sigma}^{-1}(y_{i}-f(x_i))
$$

arises naturally from multivariate Gaussian theory. Here $f(x)$ is the vector function $(f_1(x), ..., f_K(x))^T$, and $y_i$ the vector of $K$ responses for observation $i$.

(Exercise 3.11) The solution is given by (3.25); $K$ separate regressions that ignore the correlations. If the $\mathbf{\Sigma}_i$ vary among observations, then the solution for $\mathbf{B}$ no longer decouples. 

> *Proof of Exercise 3.11.*

(Section 3.7 pursue the multiple outcome problem.)

## Subset Selection

### Best-Subset Selection
### Forward- and Backward-Stepwise Selection
### Forward-Stagewise Regression

*Example: Prostate Cancer (Continued) (Page 61).*