---
layout: post
title: "Mathematics for Machine Learning: PCA"
---

**the content here is almost from the [coursera](https://www.coursera.org/learn/pca-machine-learning) with same name, the videos there can be viewed free.**
**so any content here violate any copyright, please contact me, i will remove**

___

# PCA: principal components analysis.

## several math concept will be used
### Variances of 1D data sets
Given a data set $$D=\left\{x_1,\ ...,\ x_N\right\},\ x_n\in R$$, we compute the variance of
the data set as

$$
V\left[D\right]=\frac{1}{N}\sum_{n=}^N\left(x_n-\mu\right)^2
$$

where $$\mu$$ is the mean value of the data set.

### Variances of higher-dimensional data sets
Given a data set $$D=\left\{x_1,\ ...,\ x_N\right\},\ x_n\in R^D$$, we compute the variance of the data set as

$$
V\left[D\right]=\frac{1}{N}\sum_{n=1}^N\left(x_n-\mu\right)\left(x_n-\mu\right)^T\ \in R^{D\times D}
$$

where $$\mu$$ is the mean value of the data set.

let's further explain this in a view from *covariance matrix* in probablity theory

from book <<概率论与数理统计>> of zju:
given n dimension variable $$( X_1,X_2,...,X_n )$$, if all level 2 mixed center square( 二阶混合中心矩 )

$$
c_{ij}=Cov\left(X_i,X_j\right)=E\left\{\left[X_i-E\left(X_i\right)\right]\left[X_j-E\left(X_j\right)\right]\right\},i,j=1,2,...n
$$

exist, then we call below matrix the *covariance matrix*. this matrix is symmetrical matrix because $$ c_{ij}=c_{ji}(i\ne j:i,j=1,2,...,n )$$

$$
C=\left[\begin{matrix}c_{11}&c_{12}&...&c_{1n}\\c_{21}&c_{22}&...&c_{2n}\\\vdots&\vdots&&\vdots\\c_{n1}&c_{n2}&\dots&c_{nn}\end{matrix}\right]
$$

the equations of V and C are actually same, consider an entry in V[D] result with row i and column j, $$e_{ij}$$

$$
e_{ij}=\frac{1}{N}\sum_{n=1}^N\left(x_{n,i}-E\left(X_{.,i}\right)\right)\left(x_{n,j}-E\left(X_{.,j}\right)\right)=c_{ij}
$$

### Projection onto 1D subspaces
Consider a vector space V with the dot product at the inner product
and a subspace U of V. With a basis vector b of U, we obtain the
orthogonal projection of any vector x P V onto U via

$$
\pi_u\left(x\right)=\lambda b,\lambda=\frac{b^Tx}{b^Tb}=\frac{b^Tx}{\parallel b \parallel^2}
$$

where λ is the coordinate of u(x) with respect to b.
since $$b^Tx$$ is scalar

$$
\pi_u\left(x\right)=\frac{b^Tx}{\parallel b\parallel^2}b=\frac{bb^T}{\parallel b\parallel^2}x
$$

The projection matrix P is

$$
P=\frac{bb^T}{b^Tb}=\frac{bb^T}{\parallel b\parallel^2}
$$

such that $$ \pi_u\left(x\right)=Px $$ for all x ∈Px

the properties of $$ \frac{bb^T}{\parallel b\parallel^2} $$

It is a square matrix, i.e., the number of columns equals the number of rows. The outer product $$bb^T$$ makes sure that the projection matrix is quadratic.
it’s symmetric, True because $$bb^T={(bb^T)}^T$$, such that the transpose of the projection matrix is the same as the projection matrix itself.

### Projection onto k-dimensional subspaces
Consider an n-dimensional vector space V with the dot product at the
inner product and a subspace U of V. With basis vectors b1, . . . , bk of
U, we obtain the orthogonal projection of any vector $$x\in V$$ onto U via

$$
\begin{align*}\pi_u\left(x\right)=B\lambda,\ \lambda=\left(B^TB\right)^{-1}B^Tx\end{align*}
$$

$$
\begin{align*}B=\left(b1\mid...\mid b_k\right)\in R^{n\times k}\end{align*}
$$

where $$\lambda$$ is the coordinate vector of $$\pi_u(x)$$ with respect to the basis
b1, . . . , bk of U.
The projection matrix P is

$$
P=B\left(B^TB\right)^{-1}B^T
$$

such that $$\pi_u\left(x\right)=Px$$

for all $$x\in V$$.

## problem setting and PCA objective

given a dataset:

$$
\begin{align*}
	X=\left\{x_1,...,x_N\right\},\ x_i\in R^D \\
	x_n=\sum_{i=1}^D\beta_{in}b_i\end{align*}
$$

the dimension count is D. $$b_i$$ are orthonormal basises.
if we assume that we use dot product as inner product of vector space. we can also write the $$\beta_{in}$$ as

$$
\beta_{in}=x_n^Tb_i \\
\begin{align*}x_n=\sum_{i=1}^D\beta_{in}b_i\ \Rightarrow\ b_i^Tx_n=bi^T\left(\sum_{i=1}^D\beta_{in}b_i\right)\\\Rightarrow\ b_i^Tx_n=\beta_{in}\ \Rightarrow\ x_n^Tb_i=\beta_{in}\end{align*}
$$

which means we can interpret $$\beta_{in}$$ to $$b_i$$ the orthogonal projection of $$x_n$$ onto the one dimensional subspace spanned by the $$i^{th}$$ basis vector

if we set B is a subspace spaned by basis 1 to M, and $$\tilde{x}$$ is the projection on this space

$$ \tilde{x}=BB^Tx $$

here $$ B^Tx $$ is called coordinate or code.

the key idea in PCA is to find a lower basis vectors, let’s say M. We assume the data is centered, means the dataset has mean zero.
here M divide the whole spaces into 2 parts and with different basis.
$$x_n$$ can be rewrite as:

$$
x_n=\sum_{i=1}^M\beta_{in}b_i+\sum_{i=M+1}^D\beta_{in}bi\ \in R^D
$$

then $$\tilde{x}$$ is the value with ignore the second part

here we say b1...bm spans the principle subspace.
Now the PCA problem setting is as follows. Assuming we have data $$x_1$$ to $$x_n$$, we want to find parameters $$\beta_{in}$$ and orthonormal basis vector $$b_i$$, such at the average squared reconstruction error is minimised.
average error:

$$
J=\frac{1}{N}\sum_{n=1}^N\parallel x_n-\tilde{x_n}\parallel^2
$$

problem targets at find the beta and basis which can minimum the error.

$$
\frac{\partial}{\partial_{\tilde{x_n}}}J=-\frac{2}{N}\left(x_n-\tilde{x_n}\right)^T
$$

now let’s find the $$\beta_{in}$$ would be when particial value is 0.

$$ \begin{align*}\frac{\partial}{\partial_{\beta_{in}}}\tilde{x_n}=b_i\ ,\ i=1,...,M\end{align*} $$

$$
\frac{\partial}{\partial_{\beta_{in}}}J=\frac{\partial J}{\partial\tilde{x_n}}\frac{\partial\tilde{x_n}}{\partial\beta_{in}}=-\frac{2}{N}\left(x_n-\tilde{x_n}\right)^Tb_i
$$

$$
\begin{align*}=-\frac{2}{N}\left(x_n-\sum_{j=1}^M\beta_{jn}b_j\right)^Tbi=-\frac{2}{N}\left(x_n^Tb_i-\beta_{in}b_i^Tb_i\right)\\=-\frac{2}{N}\left(x_n^Tb_i-\beta_{in}\right)\\=0\\\Leftrightarrow\beta_{in}=x_n^Tb_i\end{align*}
$$

the sum operator is simplified for in orthonomal space

$$ \beta_i^T\beta_j=0,\ i\ \ne j $$

continue the loss function and projection vector

$$
\begin{align*}\tilde{x_n}=\sum_{j=1}^M\beta_{jn}b_j\\=\sum_{j=1}^M\left(x_n^Tb_j\right)b_j\\\because x_n^Tb_j\ \ is\ a\ scalar\\=\left(\sum_{j=1}^Mb_jb_j^T\right)x_n\end{align*}
$$

here $$ \sum_{j=1}^Mb_jb_j^T $$ is the projection matrix for x on space spanced by b_1 to b_M

$$
x_n=\left(\sum_{j=1}^Mb_jb_j^T\right)x_n+\left(\sum_{j=M+1}^Db_jb_j^T\right)x_n
$$

$$
\begin{align*}x_n-\tilde{x_n}=\left(\sum_{j=m+1}^Db_jb_j^T\right)x_n=\sum_{j=m+1}^Db_jb_j^Tx_n\\\because b_j^Tx_n\ is\ a\ scalar\\=\sum_{j=M+1}^D\left(b_j^Tx_n\right)b_j\end{align*}
$$

$$
\begin{align*}J=\frac{1}{N}\sum_{n=1}^N\parallel x_n-\tilde{x_n}\ \parallel^2\\=\frac{1}{N}\sum_{n=1}^N\parallel\sum_{j=M+1}^D\left(b_j^Tx_n\right)b_j\ \parallel^2\\\because b_i\ and\ b_j\ are\ orthonomal\ basis,\ b_i\times b_j\ =0,\ i\ \ne j\\=\frac{1}{N}\sum_{n=1}^N\sum_{j=M+1}^D\left(b_j^Tx_n\right)^2\\=\frac{1}{N}\sum_n^{ }\sum_j^{ }b_j^Tx_nx_n^Tb_j\\=\sum_{j=M+1}^Db_j^T\left(\frac{1}{N}\sum_{n=1}^Nx_nx_nT\right)_Sb_j\\:\ here\ notice\ S\ is\ the\ covariance\ of\ X\\=\sum_{j=M+1}^Db_j^TSb_j=trace\left(\left(\sum_{j=M+1}^Db_jb_j^T\right)_PS\right)\end{align*}
$$

the part P here can be treated as a projection matrix of covariance matrix on subspace.
now we can rewrite our loss function using the data covariance matrix.
This projection matrix takes our data covariance matrix and project it onto the orthogonal  complement of the principal subspace.
That means, we can reformulate the loss function as  the variance of the data projected onto the subspace that we ignore. Therefore, minimising this loss is equivalent to minimising the variance of the data that lies in the subspace that is a orthogonal to the principal subspace.
 In other words, we are interested in retaining as much variance after projection as possible.  
The reformulation of the average squared reconstruction error in terms of the data covariance gives us an easy way to find the basis vector of the principal subspace.

## Finding the basis vectors that span the principal subspace

let’s firstly see when M=2, then $$b_1$$ and $$b_2$$ are orthonomize to each other. then what the value of J would be.
let $$b_1$$ be the priciple basis. then:

$$
J=b_2^TSb_2\ ,\ when\ b_2^Tb_2=1
$$

to solve this optimisation problem, we write down the [Lagrangian](https://en.wikipedia.org/wiki/Lagrange_multiplier), where lambda is the Lagrange multiplier.

$$
L=b_2^TSb_2+\lambda\left(1-b_2^Tb_2\right)
$$

$$
\frac{\partial L}{\partial\lambda}=1-b_2^Tb_2=0\Leftrightarrow b_2^Tb_2=1
$$

$$
\frac{\partial L}{\partial b_2}=2b_2^TS-2\lambda b_2^T=0\ \Leftrightarrow\ Sb_2=\lambda b_2
$$

Here we end up with an eigenvalue problem. $$b_2$$ is an eigenvector of the data covariance matrix and the Lagrange multiplier plays the role of the corresponding eigenvalue.
If we now go back to our loss function, we can use this expression.

$$
J=b_2^TSb_2=b_2^Tb_2\lambda=\lambda
$$

the average squared reconstruction error is minimised if $$\lambda$$ is the smallest eigenvalue of the data covariance matrix. And that means we need to choose $$b_2$$ as the corresponding eigenvector and that one will span the subspace that we will ignore. $$b_1$$ which spans the principle subspace is then the eigenvector that belongs to the largest eigenvalue of the data covariance matrix.
Keep in mind that the eigenvectors of the covariance matrix are already orthogonal to each other because of the symmetry of the covariance matrix.
let’s goto the general case with M basis.

$$
\begin{align*}b_j\ ,\ j=M+1,...,D\\Sb_j=\lambda_jb_j\ \\J=\sum_{j=M+1}^D\lambda_j\end{align*}
$$

Also in the general case, the average reconstruction error is minimised if we choose the basis vectors that span the ignored subspace to be the eigenvectors of the data covariance matrix that belong to the smallest eigenvalues. This equivalently means that the principal subspace is spanned by the eigenvectors belonging to the M largest eigenvalues of the data covariance matrix. This nicely aligns with properties of the covariance matrix. The eigenvectors of the covariance matrix are orthogonal to each other because of symmetry and the eigenvector belonging to the largest eigenvalue points in the direction of the data with the largest variance and the variance in that direction is given by the corresponding eigenvalue. Similarly, the eigenvector belonging to the second largest eigenvalue points in the direction of the second largest variance of the data and so on.

## PCA in high dimensions with better performance
In order to do PCA, we need to compute the data covariance matrix. In D dimensions the data covariance matrix is a D by D matrix. If D is very high, so in very high dimensions,  
then computing the eigenvalues and eigenvectors of this matrix can be quite expensive.
It scans cubic in the number of rows and columns, in this case, the number of dimensions.  
In this video, we provide a solution to this problem for the case that we have substantially fewer data points than dimensions.

$$
x_1,\ ...\ ,\ x_n\in R^D
$$

assume that the data is centered so it has means zero.

$$
\begin{align*}S=\frac{1}{N}X^TX\ \\X=\begin{bmatrix}x_1^T\\...\\x_N^T\end{bmatrix}\in R^{N\times D}\end{align*}
$$
  
We now assume that N is significantly smaller than D. That means a number of data points is significantly smaller than the dimensionality of the data. And then the rank of the covariance matrix is N. So rank of S equals N. And that also means it has D minus N plus 1 many eigenvalues which are zero. That means that the matrix is not full rank, and the rows and columns are linearly dependent. In other words, there are some redundancies. In the next few minutes, we'll exploit this and turn the D by D covariance matrix S into a full rank N by N covariance matrix without eigenvalue zero.
now let’s show how to caculate eigenvalue and eigenvector of S:

$$
\begin{align*}Sb_i=\lambda_ib_i\\\Rightarrow\ \frac{1}{N}X^TXb_i=\lambda_ib_i\\\Rightarrow\ \frac{1}{N}XX^TXb_i=\lambda_iXb_i\\now\ \frac{1}{N}XX^T\in R^{N\times N},\ take\ Xb_i\ as\ a\ new\ variable\ c_i\end{align*}
$$

this means that $$\frac{1}{N}XX^T$$ has the same non-zero eigenvalues as our target data covariance matrix but this is now an N by N matrix, so that we can compute the eigenvalues and eigenvectors much quicker than the original data covariance matrix.
then, we can compute eigenvectors now, if we left multiply above expression by X transpose:

$$
\left(\frac{1}{N}X^TX\right)X^Tc_i=\lambda_iX^Tc_i
$$

now our S matrix is recovered again. This is S and this also means that we recover X transpose times c_i is an eigenvector of S that belongs to the eigenvalue lambda.

## PCA steps
[Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)
