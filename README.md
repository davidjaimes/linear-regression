---
layout: ../layouts/post-layout.astro
date: "Mar 31, 2023"
title: Fitting a Line to Data with Linear Algebra
path: /linear-regression
badges:
    - Python
    - Astropy
image: exercise1.png
alt: exercise1
excerpt: Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data. 
---
<br>

`Linear regression` attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. For example, a modeler might want to relate the weights of individuals to their heights using a linear regression model.

## Data
```bash
 n    x    y  sig_y  sig_x  rho_xy
 1  201  592     61      9   -0.84
 2  244  401     25      4    0.31
 3   47  583     38     11    0.64
 4  287  402     15      7   -0.27
 5  203  495     21      5   -0.33
 6   58  173     15      9    0.67
 7  210  479     27      4   -0.02
 8  202  504     14      4   -0.05
 9  198  510     30     11   -0.84
10  158  416     16      7   -0.69
11  165  393     14      5    0.30
12  201  442     25      5   -0.46
13  157  317     52      5   -0.03
14  131  311     16      6    0.50
15  166  400     34      6    0.73
16  160  337     31      5   -0.52
17  186  423     42      9    0.90
18  125  334     26      8    0.40
19  218  533     16      6   -0.78
20  146  344     22      5   -0.56
```

## Ordinary Least Squares (OLS)
You have a set of $N > 2$ points ($x_i, y_i)$, with no  Gaussian uncertainties in the $x$ and $y$ directions. That is:
1. Given set of 16 points ($x_i$, $y_i$) from 5 to 20,
2. Find the best-fit line, $\hat{y_i} = b + mx_i$, and
3. Such that the sum of squared errors, $\Sigma (y_i - \hat{y_i})^2$, is minimized.
<br><br>

Constructing the matrices to solve the matrix equation, $Y = XB$:

$$
\begin{equation}
Y =
\begin{bmatrix}
    y_5 \\
    y_6 \\
    ... \\
    y_{20}
\end{bmatrix}
\\ ,
X =
\begin{bmatrix}
    1 & x_5 \\
    1 & x_6 \\
    ...& ... \\
    1 & x_{20}
\end{bmatrix}
\end{equation}
$$

$$
\begin{equation}
B =
\begin{bmatrix}
    b \\ m
\end{bmatrix}
= (X^T X)^{-1} X^T Y
\end{equation}
$$


```python
# Ordinary Linear Least Squares
df = pandas.read_csv('table1.dat', sep=' ', usecols=[1, 2])
df = df[4:]
n = len(df)

Y = df.y.values.reshape((n, 1))
X = numpy.matrix([numpy.ones(n), df.x]).T
inv = numpy.linalg.inv
B = (inv(X.T @ X)) @ (X.T @ Y)
b, m = B.item(0), B.item(1)
```

![exercise1](/exercise0.png)

## Generalized Least Squares (GLS)
You have a set of $N > 2$ points ($x_i, y_i)$, with known Gaussian uncertainties $\sigma_{yi}$ in the $y$ direction, and no uncertainty at all (that is, perfect knowledge) in the $x$ direction. You want to find the function $f(x)$ of the form

$$
\begin{equation}
f(x) = b + mx
\end{equation}
$$

where $m$ is the slope and $b$ is the intercept, that best fits the points. This exercise is adapted from `Data analysis recipes: Fitting a model to data` by David W. Hogg and Jo Bovy and Dustin Lang, 2010, 1008.4686, arXiv, astro-ph.IM, [https://arxiv.org/abs/1008.4686](https://arxiv.org/abs/1008.4686).

Using the covariance matrix, $C$, prepare the matrices to solve the matrix equation, $Y = XB$:

$$
\begin{equation}
C =
\begin{bmatrix}
    \sigma^2_{y1} & 0 & ... & 0 \\
    0 & \sigma^2_{y2} & ... & 0 \\
    ...& ... & ... & ... \\ 
    0 & 0 & ... & \sigma^2_{yN}
\end{bmatrix}
\end{equation}
$$

$$
\begin{equation}
B = (X^T C^{-1} X)^{-1} (X^T C^{-1} Y)
\end{equation}
$$

$$
\begin{equation}
V = 
\begin{bmatrix}
    \sigma^2_b & \alpha \\
    \alpha & \sigma^2_m
\end{bmatrix}
\end{equation}
$$

$$
\begin{equation}
V = (X^T C^{-1} X)^{-1}
\end{equation}
$$

<br/>

## Exercise 1

Using the standard linear algebra method of this Section, fit a straight line $y = mx + b$ to the $x$, $y$, $\sigma_y$ values for data points 5 through 20 in Table 1 on page 6. That is, ignore the first four data points, and also ignore the columns for $\sigma_x$ and $\rho_{xy}$. Make a plot showing the points, their uncertainties, and the best-fit line. Your plot should end up looking like Figure 1. What is the standard uncertainty variance $\sigma^2_m$ on the slope of the line?

```python
# Generalized Linear Least Squares
df = pandas.read_csv('table1.dat', sep=' ', usecols=[1,2,3])
df = df[4:]
n = len(df)

Y = df.y.values.reshape((n, 1))
X = numpy.matrix([numpy.ones(n), df.x]).T
C = numpy.diag(pow(df.sig_y.values, 2))
inv = numpy.linalg.inv
b1 = inv(X.T @ inv(C) @ X)
b2 = X.T @ inv(C) @ Y
B = b1 @ b2
b, m = B.item(0), B.item(1)
sig_b = numpy.sqrt(b1.item(0))
sig_m = numpy.sqrt(b1.item(3))
```

![exercise1](/exercise1.png)
