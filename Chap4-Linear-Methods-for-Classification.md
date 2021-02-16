<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script> -->

# **4 Linear Methods for Classification**

## **4.1 Introduction**

This chapter focos on linear method for classification (the decision boundaries are linear). Suppose predictor $G(x)$ takes values in a distrete set $\mathcal{G}$, we can always divide the input space into a collection of regions labeled according to the classification. 

Suppose, there are $K$ classes, for convenience labeled $1,2,...,K$. There are several different ways in which linear decision boundaries can be found.

Given a model discriminant functions $\delta_k(x)$ for each class, and classify $x$ to the class with the largest value for its discriminant function. The regression approach and that model the posterior probabilities $\Pr(G=k|X=x)$ are the members of this class of methods as soon as the monotone transformation of $\delta_k(x)$ and $\Pr(G=k|X=x)$ are linear for the decision boundaries to be linear. 

- The regression approach: Supppose the fitted linear model for the $k$th indicator response variable is $\hat{f}_k(x)=\hat{\beta}_{k0}+\hat{\beta}_k^Tx$. The decision boundary between class $k$ and $l$ is that set of points for which $\hat{f}_k(x)=\hat{f}_{\ell}(x)$, that is the set $\{x:(\hat{\beta}_{k0}-\hat{\beta}_{\ell 0})+(\hat{\beta}_k-\hat{\beta}_{\ell})^Tx = 0\}$, an affine set or hyperplane.
- The posterior probilities: The common useful model is 
  $$\tag{4.1}
  \begin{aligned}
    \Pr(G=k|X=x) &= \frac{\exp(\beta_{k0}+\beta_k^Tx)}{1+\sum_{\ell=1}^{K-1}\exp(\beta_{\ell 0}+\beta_{\ell}^Tx)}, k = 1,..., K-1\\
     \Pr(G=K|X=x) &= \frac{1}{1+\sum_{\ell=1}^{K-1}\exp(\beta_{\ell 0}+\beta_{\ell}^Tx)}.
  \end{aligned}
  $$
  Here the monotone transformation is the *logit* trainsformation and in fact we see that 
  $$\tag{4.2}
  \log\frac{\Pr(G=k|X=x)}{\Pr(G=\ell|X=x)} = (\beta_{k0}-\beta_{\ell0})+(\beta_{k}-\beta_{\ell})^Tx.
  $$
  The decision boundary is the set of points for which the *log-odds* are zero. We discuss two very popular but different methods that result in linear log-odds or logits: linear discriminant analysis and linear logistic regression. Although they differ in their derivation, the essential difference between them is in the way the linear function is fit to the training data.
- Explicitly model the boundaries between the classes as linear: The first is the well-known *perceptron*, the second is *optimally separating hyperplane*. We treat the separable case here, and defer treatment of the nonseparable case to Chapter 12.

While this entire chapter is devoted to linear decision boundaries, there is considerable scope for generalization. For example, expand the variable set by including their squares and cross-products. This approach can be used with any basis transformation $h(X)$ where $h:\mathbb{R}^p\to \mathbb{R}^q$ with $q > p$.

## **4.2 Linear Regression of an Indicator Matrix**

Here each of the response categories are coded via an indicator variable. Then if $\mathcal{G}$ has $K$ classes, there will be $K$ such indicators $Y_k, k=1,..., K$ with $Y_k=1$ if $G=k$ else $0$. These are collected together in a vector $Y=(Y_1,..., Y_K)$, and the $N$ training instances of there form an $N\times K$ *indicator response matrix* $\mathbf{Y}$ (0-1 matrix, each row having a single 1). We fit a linear regression model to each columns of $\mathbf{Y}$ simultaneously, and the fit is given by 
$$\tag{4.3}
\hat{\mathbf{Y}}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}.
$$
Note that 
  - the coefficient vector for each response column $\mathbf{y}_k$;
  - the $\hat{\mathbf{B}}= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$ has $(p+1)\times K$ coefficients;
  - compute the fitted output $\hat{f}(x)^T= (1,x^T)\hat{\mathbf{B}}$, a $K$ vector;
  - identify the largest component and classify accordingly:
  $$\tag{4.4}
  \hat{G}(x)=\argmax_{k\in \mathcal{G}}\hat{f}_k(x).
  $$

What is the rationable for this approach? We know that the regression as an estimate of conditional expectation. So, for the random variable $Y_k$, $E(Y_k|X=x)=\Pr(G=k|X=x)$, so conditional expectation of each of the $Y_k$ seems a sensible goal. However, the $\hat{f}_k(x)$ can not be reasonable estimates of the posterior probabilities $\Pr(G=k|X=x)$ (could be negative or greater than 1). In fact on many problems it gives similar results to more standard linear methods for classification. If we allow linear regression onto basis expansions $h(X)$ of the inputs, this approach can lead to consistent estimates of the probabilities. As the size of the training set $N$ grows bigger, we adaptively include more basis elements so that linear regression onto these basis functions approaches conditional expectation. We discuss such approaches in Chapter 5.

A more simplistic viewpoint is to construct targets $t_k$ for each class, where $t_k$ is the $k$th column of the $K\times K$ identity matrix. With the same coding as before, the response vector $y_i$ ($i$th row of $\mathbf{Y}$) for observation $i$ has the value $y_k=t_k$ if $g_i=k$. We might then fit the linear model by least squares:
$$\tag{4.5}
\min_{\mathbf{B}}\sum_{i=1}^N\|y_i-[(1,x_i^T)\mathbf{B}]^T\|^2.
$$
The criterion is a sum-of-squared Euclidean distances of the fitted vectors from their targets. A new observation is classified by computing its fitted vector $\hat{f}(x)$ and classifying to the cloest target:
$$\tag{4.6}
\hat{G}(x) = \argmin_k\|\hat{f}(x)-t_k\|^2.
$$
This is exactly the same as the previous approach:
   - The closest target classification rule (4.6) is easily seen to be exactly the same as the maximum fitted component criterion (4.4).

There is a serious problem with the regression approach when the number of classes $K \geq 3$, especially prevalent when $K$ is large. Because of the rigid nature of the regression model, classes can be masked by others. See figure 4.2 and 4.3.

<div align=center>
<img src="pic/figure4.2.png" width="61.8%">
</div>

<div align=center>
<img src="pic/figure4.3.png" width="61.8%">
</div>

For the cases in figure 4.3, if there are $K\geq 3$ classes are lined up, polynomial terms up to degress $K-1$ might be needed to resolve them. So in $p$-dimensional input space, one would need general polynomial terms and cross-products of total degree $K − 1$, $O(p^{K−1})$ terms in all, to resolve such worst-case scenarios. The example is extreme, but for large $K$ and small $p$ such maskings naturally occur. As a more realistic illustration, Figure 4.4 is a projection of the training data for a vowel recognition problem onto an informative two-dimensional subspace.

<div align=center>
<img src="pic/figure4.4.png" width="61.8%">
</div>

<div align=center>
<img src="pic/table4.1.png" width="61.8%">
</div>

## **4.3 Linear Discriminant Analysis**

Decision theory for classification (Section 2.4) tells us that we need to know the class posteriors $\Pr(G|X)$ for optimal classification. Suppose $f_k(x)$ is the class-conditional density of $X$ in class $G=k$, and let $\pi_k$ be the prior probability of class $k$, with $\sum_{k=1}^K\pi_k=1$. A simple application of Bayes theorem gives us
$$\tag{4.7}
\Pr(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_{\ell=1}^Kf_{\ell}(x)\pi_{\ell}}.
$$
We see that in terms of ability to classify, having the $f_k(x)$ is almost equivalent to having the quantity $\Pr(G = k|X = x)$. 

Many techniques are based on models for the class densities:
  - linear and quadratic discriminant analysis use Gaussian densities;
  - more flexible mixtures of Gaussians allow for nonlinear decision boundaries (Section 6.8);
  - general nonparametric density estimates for each class density allow the most flexibility (Section 6.6.2);
  - Naive Bayes models are a variant of the previous case, and assume that each of the class densities are products of marginal densities; that is, they assume that the inputs are conditionally independent in each class (Section 6.6.3).

Suppose that we model each class density as multivariate Gaussian
$$\tag{4.8}
f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}}e^{-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)}.
$$

Linear discriminant analysis (LDA) arises in the special case when we assume that the classes have a common covariance matrix $\Sigma_k=\Sigma \quad \forall k$. In comparing two classes $k$ and $\ell$, it is sufficient to look at the log-ratio, and we see that
$$\tag{4.9}
\begin{aligned}
\log \frac{\Pr(G=k|X=x)}{\Pr(G=\ell|X=x)} &= \log\frac{f_k(x)}{f_{\ell}(x)}+\log\frac{\pi_k}{\pi_{\ell}}\\
&=\log\frac{\pi_k}{\pi_{\ell}}-\frac{1}{2}(\mu_k+\mu_{\ell})^T\Sigma^{-1}(\mu_k-\mu_{\ell})+x^T\Sigma^{-1}(\mu_k-\mu_{\ell}),
\end{aligned}
$$
an equation linear in $x$. For any pair of classes $k,\ell$, the decision boundary is the set where $\Pr(G=k|X=x)=\Pr(G=\ell|X=x)$ is linear in $x$; in $p$ dimensions a hyperplane. Figure 4.5 shows an example with three classes from three Gaussian distributions with a common covariance matrix and $p=2$. 

<div align=center>
<img src="pic/figure4.5.png" width="61.8%">
</div>

Notice that the decision boundaries are not the perpendicular bisectors of the line segments joining the centroids. This (perpendicular) would be the case if the covariance $\Sigma$ were spherical $\sigma^2\mathbf{I}$, and the class priors were equal. From (4.9) we see that *linear discriminant functions*

$$\tag{4.10}
\delta_k(x) = x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+\log \pi_k
$$
are an equivalent description of the decision rule, with $G(x)=\argmax_k\delta_k(x)$.

In practice we do not know the parameters of the Gaussian distributions, and will need to estimate them using our training data:
  - $\hat{\pi}_k=N_k/N$, where $N_k$ is the number of class-$k$ observations;
  - $\hat{\mu}_k=\sum_{g_i=k}x_i/N_k$;
  - $\hat{\Sigma} = \sum_{k=1}^K\sum_{g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T/(N-K)$.

**With two classes**, the LDA rule classifies to class 2 if 
$$\tag{4.11}
x^T\hat{\Sigma}^{-1}(\hat{\mu}_2-\hat{\mu}_1)>\frac{1}{2}(\hat{\mu}_2+\hat{\mu}_1)^T\Sigma^{-1}(\hat{\mu}_2-\hat{\mu}_1)-\log(N_2/N_1).
$$
Suppose we code the targets in the two classes as +1 and −1, respectively. It is easy to show that the coefficient vector from least squares is proportional to the LDA direction given in (4.11) (Exercise 4.2). [In fact, this correspondence occurs for any (distinct) coding of the targets; see Exercise 4.2]. However unless $N_1 = N_2$ the intercepts are different and hence the resulting decision rules are different.

> Exercise 4.2.

Since this derivation of the LDA direction via least squares does not use a Gaussian assumption for the features, its applicability extends beyond the realm of Gaussian data. However the derivation of the particular intercept or cut-point given in (4.11) does require Gaussian data. Thus it makes sense to instead choose the cut-point that empirically minimizes training error for a given dataset. This is something we have found to work well in practice, but have not seen it mentioned in the literature.

**With more than two classes**, LDA is not the same as linear regression of the class indicator matrix, and it avoids the masking problems associated with that approach (Hastie et al., 1994). A correspondence between regres- sion and LDA can be established through the notion of optimal scoring, discussed in Section 12.5.

Getting back to the general discriminant problem (4.8), if the $\Sigma_k$ are not assumed to be equal, then the convenient cancellations in (4.9) do not occur, we then get *quadratic discriminant functions* (QDA),
$$\tag{4.12}
\delta_k(x) = -\frac{1}{2}\log|\Sigma_k|-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\log\pi_k.
$$
The decision boundary between each pair of classes $k$ and $\ell$ is described by a quadratic equation $\{x : \delta_k(x) = \delta_{\ell}(x)\}$.

Figure 4.6 shows an example (from Figure 4.1 on page 103) where the three classes are Gaussian mixtures (Section 6.8) and the decision boundaries are approximated by quadratic equations in $x$. 

<div align=center>
<img src="pic/figure4.6.png" width="61.8%">
</div>

> For this figure and many similar figures in the book we compute the decision bound- aries by an exhaustive contouring method. We compute the decision rule on a fine lattice of points, and then use contouring algorithms to compute the boundaries.

When $p$ is large this can mean a dramatic increase in parameters. For LDA, it seems there are $(K-1)\times(p+1)$ parameters, while for QDA there will be $(K-1)\times\{p(p+1)/2+p+1\}$ parameters. Both LDA and QDA perform well on an amazingly large and diverse set of classification tasks. The question arises why LDA and QDA have such a good track record. The reason is not likely to be that the data are approximately Gaussian, and in addition for LDA that the covariances are approximately equal. **More likely a reason is that the data can only support simple decision boundaries such as linear or quadratic, and the estimates provided via the Gaussian models are stable.** This is a bias variance tradeoff—we can put up with the bias of a linear decision boundary because it can be estimated with much lower variance than more exotic alternatives. This argument is less believable for QDA, since it can have many parameters itself, although perhaps fewer than the non-parametric alternatives.

### **4.3.1 Regularized Discriminant Analysis**

Friedman (1989) proposed a compromise between LDA and QDA, which allows one to shrink the separate covariances of QDA toward a common covariance as in LDA. These methods are very similar in flavor to ridge regression. The regularized covariance matrices have the form
$$\tag{4.13}
\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k+(1-\alpha)\hat{\Sigma},
$$
where $\alpha\in [0,1]$ and $\hat{\Sigma}$ is the pooled covariance matrix as used in LDA. In practice $\alpha$ can be chosen based on the performance of the model on validation data, or by cross-validation.

Figure 4.7 shows the results of RDA applied to the vowel data.
<div align=center>
<img src="pic/figure4.7.png" width="61.8%">
</div>
Similar modifications allow $\hat{\Sigma}$ itself to be shrunk toward the scalar covariance,

$$\tag{4.14}
\hat{\Sigma}(\gamma) = \gamma\hat{\Sigma}+(1-\gamma)\hat{\sigma}^2\mathbf{I}
$$

for $\gamma\in [0,1]$. Replacing $\hat{\Sigma}$ in (4.13) by $\hat{\Sigma}(\gamma)$ leads to a more general family of covariances $\hat{\Sigma}(\alpha, \gamma)$ indexed by a pair of parameters.

In Chapter 12, we discuss other regularized versions of LDA, which are more suitable when the data arise from digitized analog signals and images. In Chapter 18 we also deal with very high-dimensional problems, where for example the features are gene- expression measurements in microarray studies. There the methods focus on the case $\gamma = 0$ in (4.14), and other severely regularized versions of LDA.

### **4.3.2 Computations for LDA**

We briefly digress on the computations required for LDA and especially QDA. By the eigen decomposition for each $\hat{\Sigma}_k=\mathbf{U_kD_kU_k}^T$, where $\mathbf{U}_k$ is $p\times p$ orthonormal and $\mathbf{D}_k$ a diagonal matrix of positive eigenvalues $d_{k\ell}$. Then the ingredients for $\delta_k(x)$ (4.12) are
  -  $(x-\hat{\mu}_k)^T\hat{\Sigma}_k^{-1}(x-\hat{\mu}_k) = [\mathbf{U}_k^T(x-\hat{\mu}_k)]^T\mathbf{D}_k^{-1}[\mathbf{U}_k^T(x-\hat{\mu}_k)]$;
  - $\log|\hat{\Sigma}_k|=\sum_{\ell}\log d_{k\ell}$.

In light of the computational steps outlined above, the LDA classifier
can be implemented by the following pair of steps:

  - *Sphere* the1data with respect to the common covariance estimate $\hat{\Sigma}: X^* \leftarrow D^{-\frac{1}{2}} U^T X$, where $\hat{\Sigma} = \mathbf{UDU}^T$. The common covariance estimate of $X^*$ will now be the identity.
  
    > Since $\hat{\Sigma} = \mathbf{UDU}^T$, 
    > $$
    \hat{\Sigma}^* = \mathbf{D}^{-\frac{1}{2}}\mathbf{U}^T\hat{\Sigma}\mathbf{U}\mathbf{D}^{-\frac{1}{2}} = \mathbf{D}^{-\frac{1}{2}}\mathbf{U}^T\mathbf{U}\mathbf{D}\mathbf{U}^T\mathbf{U}\mathbf{D}^{-\frac{1}{2}} = \mathbf{I}.

    > $$

  - Classify to the closest class centroid in the transformed space, modulo the effect of the class prior probabilities $\pi_k$.
    >  $$\delta^*_k(x)=-\frac{1}{2}\|x^*-\hat{\mu}^*_k\|_2^2+\log \pi_k$$ 

### **4.3.3 Reduced-Rank Linear Discriminant Analysis**

LDA as a restricted Gaussian classifier allow us to view informative low-dimensional projections of the data. The $K$ centroids in $p$-dimensional input space lie in an affine subspace of dimension $\leq K-1$, and if $p$ is much larger than $K$, this will be a considerable drop in dimension. Moreover, in locating the closest centroid, we can ignore distances orthogonal to this subspace, since they will contribute equally to each class. Thus there is a fundamental dimension reduction in LDA, namely, that we need only consider the data in a subspace of dimension at most $K − 1$.

We might then ask for a $L< K-1$ dimensinal subspace $H_L\subset H_{K-1}$ optimal for LDA in some sense. Fisher defined optimal to mean that the projected centroids were spread out as much as possible in terms of variance. This amounts to finding principal component subspaces of the centroids themselves. 

Figure 4.4 shows such an optimal two-dimensional subspace for the vowel data. Here there are eleven classes, each a different vowel sound, in a ten-dimensional input space. The centroids require the full space in this case, since $K − 1 = p$, but we have shown an optimal two-dimensional subspace.  The dimensions are ordered, so we can compute additional dimensions in sequence.

Figure 4.8 shows four additional pairs of coordinates, also known as canonical or discriminant variables.
<div align=center>
<img src="pic/figure4.8.png" width="61.8%">
</div>

In summary, finding the sequences of optimal subspaces for LDA involves the following steps:
  - compute the $K\times p$ matrix of class centroids $\mathbf{M}$ and the common covariance matrix $\mathbf{W}$ (for within-class covariance);
  - compute $\mathbf{M}^*=\mathbf{M}\mathbf{W}^{-\frac{1}{2}}$ using the eigen-decomposition of $\mathbf{W}$;
  - compute $\mathbf{B}^*$, the covariance matrix of $\mathbf{M}^*$ ($\mathbf{B}$ for between-class covariance), and its eigen-decomposition $\mathbf{B}^*=\mathbf{V}^*\mathbf{D}_{B}\mathbf{V}^{*T}$. The columns $v_{\ell}^*$ of $\mathbf{V}^*$ in sequence from first to last define the coordinates of the optimal subspaces.

Combining all these operations the $\ell$th discriminant variable is given by $Z_{\ell}=v_{\ell}^TX$ with $v_{\ell}=\mathbf{W}^{-\frac{1}{2}}v_{\ell}^*$.

Fisher arrived at this decomposition via a different route, without referring to Gaussian distributions at all. He posed the problem:

*Find the linear combination $Z = a^TX$ such that the between-class variance is maximized relative to the within-class variance.*

> The between class variance matrix is the variance of the class means of $X$, 
> $$
\mathbf{B} = \sum_{k=1}^{K}\sum_{g_{i}=k}{(\hat{\mu}_{k}-\hat{\mu})(\hat{\mu}_{k}-\hat{\mu})^{T}/(K-1)},
> $$
>
>
> the within class variance is the pooled variance about the means
> $$
\mathbf{W} = \sum_{k=1}^{K}\sum_{g_{i}=k}{(x_{i}-\hat{\mu}_{k})(x_{i}-\hat{\mu}_{k})^{T}/(N-K)},
> $$
> The total covariance matrix of $X$, ignoring class information
> $$
\mathbf{T} = \sum_{k=1}^K\sum_{g_i=k}(x_i-\hat{\mu})(x_i-\hat{\mu})^T/(N-1).
> $$
> It is easy to prove that $\mathbf{T=B+W}$.
> The between-class variance of $Z$ is $a^T\mathbf{B}a$ and the within-class variance is $a^T\mathbf{W}a$.

Figure 4.9 shows why this criterion makes sense. 
<div align=center>
<img src="pic/figure4.9.png" width="61.8%">
</div>

Fisher's problem therefore amounts to maximizing the *Rayleigh quotient*,
$$\tag{4.15}
\max_a\frac{a^T\mathbf{B}a}{a^T\mathbf{W}a},
$$
or equivalently 
$$\tag{4.16}
\max_a a^T\mathbf{B}a \text{ subject to } a^T\mathbf{W}a=1.
$$
> which can rewrite after the convenient basis change $a^* = \mathbf{W}^{1/2}a$, $a^T\mathbf{B}a = a^{*T}\mathbf{W}^{-1/2}\mathbf{B}\mathbf{W}^{-1/2}a^* = a^{*T}\mathbf{B}^*a^*$, 
> $$
\min_{a^*}-\frac{1}{2}a^{*T}\mathbf{B}^*a^* \text{ subject to } a^{*T}a^*=1.
> $$
> The Lagrangien for this problem writes
> $$
L=-\frac{1}{2}a^{*T}\mathbf{B}^*a^*+\frac{1}{2}\lambda(a^{*T}a^*-1).
> $$
> and the Karush-Kuhn-Tucker conditions give
> $$
\mathbf{B}^*a^*=\lambda a^* \equiv \mathbf{W}^{-1}\mathbf{B}a=\lambda a.
> $$
> Thus, the optimal $a^*$ corresponding the largest eigenvalue of $\mathbf{B}^*$, that is  $v_{1}^*$. And $a$ given by the largest eigenvalue of $\mathbf{W}^{-1}\mathbf{B}$. Similarly one can find the next direction $v_{2}^*$, and so on. $v_{\ell}=\mathbf{W}^{-\frac{1}{2}}v_{\ell}^*$.
>  It is not hard to show (Exercise 4.1) that the optimal $a_1$ is identical to $v_1$ defined above. 

The $a_{\ell}$ are referred to as discriminant coordinates, not to be confused with discriminant functions. They are also referred to as *canonical variates*, since an alternative derivation of these results is through a canonical correlation analysis of the indicator response matrix $\mathbf{Y}$ on the predictor matrix $\mathbf{X}$. This line is pursued in Section 12.5.

To summarize the developments so far:
  - Gaussian classification with common covariances leads to linear deci- sion boundaries. Classification can be achieved by sphering the data with respect to $\mathbf{W}$, and classifying to the closest centroid (modulo $\log\pi_k$) in the sphered space.
  - Since only the relative distances to the centroids count, one can confine the data to the subspace spanned by the centroids in the sphered space.
  - This subspace can be further decomposed into successively optimal subspaces in term of centroid separation. This decomposition is identical to the decomposition due to Fisher.

One can show that this is a Gaussian classification rule with the additional restriction that the centroids of the Gaussians lie in a $L$-dimensional subspace of $\mathbb{R}^p$. Fitting such a model by maximum likelihood, and then constructing the posterior probabilities using Bayes’ theorem amounts to the classification rule described above (Exercise 4.8).
> Exercise 4.8

Gaussian classification dictates the logπk correction factor in the dis- tance calculation. The reason for this correction can be seen in Figure 4.9. The misclassification rate is based on the area of overlap between the two densities. If the $\pi_k$ are equal (implicit in that figure), then the optimal cut-point is midway between the projected means. If the $\pi_k$ are not equal, moving the cut-point toward the smaller class will improve the error rate. As mentioned earlier for two classes, one can derive the linear rule using LDA (or any other method), and then choose the cut-point to minimize misclassification error over the training data.

Figure 4.10 shows the results. Figure 4.11 shows the decision boundaries for the classifier based on the two-dimensional LDA solution.

<div align=center>
<img src="pic/figure4.10.png" width="61.8%">
</div>

<div align=center>
<img src="pic/figure4.11.png" width="61.8%">
</div>

There is a close connection between Fisher’s reduced rank discriminant analysis and regression of an indicator response matrix. It turns out that LDA amounts to the regression followed by an eigen-decomposition of $\hat{\mathbf{Y}}^T\mathbf{Y}$.  In the case of two classes, there is a single discriminant variable that is identical up to a scalar multiplication to either of the columns of $\hat{\mathbf{Y}}$. These connections are developed in Chapter 12. A related fact is that if one transforms the original predictors $\mathbf{X}$ to $\hat{\mathbf{Y}}$ , then LDA using $\hat{\mathbf{Y}}$ is identical to LDA in the original space (Exercise 4.3).
> Exercise 4.3.

## **4.4 Logistic Regression**