# Machine Learning Algorithms

Machine learning is a burgeoning field of computer science that essentially allows computers to analyze patterns in order to "learn" how to make predictions or classifications of input based on parameters learned from previous "experience". Researchers in this field have been furthering the frontier of artificial intelligence to the point that our computers can now recognize and read text from photos, process grammatical structures better than most humans (see Google's [Parsey McParseface](https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html)), and much more!

In my machine learning course on Coursera's online learning platform, I had the opportunity to build many of the core algorithms upon which many AIs are built. The first group of algorithms listed below are "supervised", meaning the programmer supplies input and output pairs for the computer to operate on to learn certain parameters. The "unsupervised" algorithms allow computers to do things like cluster data into discrete groups and reduce data dimensionality. Read on to find out just how amazing and powerful these algorithmic concepts are!

## Optimization and Regression

One of the most fundamental genera of algorithms needed to build a successful learning machine is the optimization function. Having a quick way to optimize an equation with many variables--usually an equation called a "cost function"--is essential to nearly all machine learning applications, and is often accomplished with gradient descent. For linear regression problems with few features, optimization may actually be done by solving the "normal equation", but in most other cases, gradient descent (though not 100% exact like the normal equation) is used for optimization for reasons discussed in the sections below.

### The Cost function

Integral to nearly all optimization problems is the idea of the "cost function", labeled *J* in my codebase. The cost function, also known as the "loss function" in other mathematical applications, is a method of analyzing the error associated with a set of parameters *θ* when used to calculate a hypothesis function *h(θ)*. In many of the applications in this codebase, *J(θ)* is calculated according to the formula:

![J(θ)][basic J],

where *m* is the number of examples in our training set. Thus, in the case of a system with one variable feature *x₁* where *h(θ)* is calculated as *h(θ) = θ₀ + θ₁x₁*, a plot of *J* with axes *θ₀* and *θ₁* will give us a paraboloid structure:

![Cost function over theta][2d theta cost]

In the case of linear regression, the function *J* may be minimized with respect to *θ* by either gradient descent or the normal equation.

### Gradient Descent

Gradient descent is a method of minimizing a cost function that calculates partial derivatives of the cost function with respect to parameters *θ* and multiplies each by a learning rate *α* to update each parameter *θᵢ* and traverse the cost function iteratively until reaching a minimum. In the case of the cost function for linear regression, for example, the partial derivatives are calculated according to the formula:

![J partial derivative, linear regression][partial grad],

and parameters are updated thus:

![Parameter update][theta update].

The name "gradient descent" makes sense by this definition, since the gradient (another name for the vectorized set of partial derivatives) is used to make small changes to the parameter vector *θ* in a direction guaranteed to make *J* smaller, assuming a small enough learning rate *α*. Graphically, it looks like the algorithm is "stepping down" the graph *J* with each parameter update until reaching a local minimum. An important point must be made that gradient descent is **not** guaranteed to find a global minimum for non-paraboloid functions *J*, and thus the algorithm is frequently repeated with different random initialization vectors *θ* to ensure a suitable optimum has been found.

### The Normal Equation

The normal equation solution to linear regression is a result that is not necessarily applicable to other cost function optimization, but is nevertheless useful in solving for optimal *θ* when the number of features *n* in the system is low (i.e., ≤1000). The formula is found by setting the gradient vector *∇J* with respect to *θ* equal to the zero vector and solving for *θ*. This gives the following equation:

![The Normal Equation][normal equation]

The reason this solution may not be plausible for systems with large numbers of features is that the calculation of the inverse of a matrix is *O(n³)* time, while the runtime for gradient descent is roughly *O(lmn)*, where *l* is the number of algorithm iterations. Furthermore, gradient descent may be allowed to terminate early, as long as the magnitude of the gradient is sufficiently small, while the normal equation must do the full calculation. As such, though the normal equation gives us an exact solution to the minimization problem, gradient descent often outperforms the normal equation speed-wise and is generally equally effective.

### Linear Regression

The first machine learning algorithm that I implemented was linear regression, which most people likely encountered early in their academic careers fiddling around in Excel. The aim of linear regression, as indicated by its name, is to fit a line to a data set.


## Classification Problems

This genus of machine learning problem is characterized by labelled output data; that is to say, the input data vector *x* is used to predict some classification *y* of the data point. The simplest of such problems only has binary classification, 0 or 1. Often, these are called "negative" and "positive" examples, respectively. Multi-class classification occurs when the output label can be one of many different classes. As we will see, this problem can be treated in a similar manner, and is thus not much more complicated than simple binary classification.

<!-- ### Logistic Regression

### Neural Networks

#### Feedforward Propagation

#### Back Propagation

### Support Vector Machines

### Troubleshooting: Bias and Variance -->


## Unsupervised Learning

Unsupervised learning algorithms operate on training sets where there is only input data with no output pair. Applications of such algorithms include anomaly detection, recommendation-making, data clustering, and data compression. This type of machine learning assumes some sort of relation inherent in the input data, attempts to find it, and uses it for the purposes listed above. This means that most algorithms must be designed with limitations in place to prevent overfitting of data or too much loss of data (in the case of dimensionality reduction). A well-designed algorithm will in turn simplify the role of the programmer significantly.

### Anomaly Detection

The detection of outliers in data sets is an extremely valuable tool, since this can flag products that may not conform to standards or flag suspicious activity. What anomaly detection algorithms do is use input data to develop a set of parameters that define a probability function *p(x)* that gives the probability that the input is an outlier. The programmer is responsible for picking a threshold *ε* for flagging outliers when *p(x) < ε*. Frequently, this threshold is selected for optimal F1 score on a cross-validation set.

<!-- ### Recommender Systems

### Clustering with k-Means

### Principal Component Analysis -->


***
*I would like to give special thanks to my Coursera teacher, Dr. Andrew Ng, for the amazing learning experience I had while implementing these projects. Note also that the images and equation snapshots used above come from project manuals posted throughout the course.*

<!-- TODO: Finish cost function, add in gradient descent (don't forget picture)
Replace θ and α ε -->

[basic J]: documentation/basic_cost_func.png "Cost function: Square Difference"
[2d theta cost]: documentation/J_lin_regr.png "Cost function plotted over θ₀ and θ₁"
[partial grad]: documentation/gradient_def.png "Gradient by partial derivative"
[theta update]: documentation/pseudocode_theta_update.png "Simultaneous update for all thetas"
[normal equation]: documentation/normal_equation.png "θ solution to normal equation"
