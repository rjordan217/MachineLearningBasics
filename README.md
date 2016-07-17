# Machine Learning Algorithms

Machine learning is a burgeoning field of computer science that essentially allows computers to analyze patterns in order to "learn" how to make predictions or classifications of input based on parameters learned from previous "experience". Researchers in this field have been furthering the frontier of artificial intelligence to the point that our computers can now recognize and read text from photos, process grammatical structures better than most humans (see [Parsey McParseface][https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html]) and much more!

In my machine learning course on Coursera's online learning platform, I had the opportunity to build many of the core algorithms upon which many AIs are built. The first group of algorithms listed below are "supervised", meaning the programmer supplies input and output pairs for the computer to operate on to learn certain parameters. The "unsupervised" algorithms allow computers to do things like cluster data into discrete groups and reduce data dimensionality. Read on to find out just how amazing and powerful these algorithmic concepts are!

## Optimization and Regression

One of the most fundamental genera of algorithms needed to build a successful learning machine is the optimization function. Having a quick way to optimize an equation with many variables--usually an equation called a "cost function"--is essential to nearly all machine learning applications, and is often accomplished with gradient descent. For linear regression problems with few features, optimization may actually be done by solving the "normal equation", but in most other cases, gradient descent (though not 100% exact like the normal equation) is used for optimization for reasons discussed in the sections below.

### The Cost function

Integral to nearly all optimization problems is the idea of the "cost function", labeled *J* in my codebase. The cost function, also known as the "loss function" in other mathematical applications, is a method of analyzing the error associated with a set of parameters *ø* when used to calculate a hypothesis function *h(ø)*. In many of the applications in this codebase, *J(ø)* is calculated according to the formula:
![J(ø)][basic J]. Thus, in the

### Gradient Descent

Gradient descent is a method of minimizing parameters that relies on

### The Normal Equation



### Linear Regression

The first machine learning algorithm that I implemented was linear regression, which most people likely encountered early in their academic careers fiddling around in Excel. The aim of linear regression, as indicated by its name, is to fit a line to a data set.


## Classification Problems

### Logistic Regression

### Neural Networks

#### Feedforward Propagation

#### Back Propagation

### Support Vector Machines

### Troubleshooting: Bias and Variance


## Unsupervised Learning

### Anomaly Detection

### Recommender Systems

### Clustering with k-Means

### Principal Component Analysis


***
*I would like to give special thanks to my Coursera teacher, Dr. Andrew Ng, for the amazing learning experience I had while implementing these projects. Note also that the images and equation snapshots used above come from project manuals posted throughout the course.*

<!-- TODO: Finish cost function, add in gradient descent (don't forget picture) -->

[basic J]: documentation/basic_cost_func.png "Cost function: Square Difference"
[partial grad]: documentation/gradient_def.png "Gradient by partial derivative"
