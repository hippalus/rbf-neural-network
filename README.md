# CMP5133-Artificial Neural Networks
#### Homework 1 - RBF Network Implementation

Using the attached regression data, you are going to train a radial basis function (RBF) network with one input, one output and H Gaussian units/experts.

#### Part 1:
a) Try at least three different H values so one underfits, one fits well and one overfits. 
Initialize using online k-means and then use gradient-descent.
For each case, after convergence, plot the p_h(RBF), weighted values w_h*p_h, and the overall output, together 
with the training data so that we can see how the fit is distributed  over the hidden units.

b) Train "Rules and Exceptions" model on the same dataset. Try at least three different H values so one underfits, 
one fits well and one overfits.
Initialize using online k-means and then use  gradient-descent. For each case, after convergence, plot the
p_h(RBF), weighted values w_h*p_h, and the overall output, together with the training data so that we can see how the fit is distributed 
over the hidden units.
Plot different values in different styles (dashed, dotted, etc) so that they can be distinguished or plot them in different colors.

#### Part2:
Plot also training and validation errors as a function of H for both cases given above; 
plot these using averages over at least 10 runs starting from different random weights.

#### Instructions

Do NOT use a built-in function or an implementation from another source but implement it yourself!


#### Submit a single PDF document including the following
a) results (plots),

b) a short report explaining what you did (e.g. how you have determined the spread values, which parameters did you use for clustering, etc.

c) discussions (compare the results in a comparative manner).
