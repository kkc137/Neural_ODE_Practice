# Neural_ODE_Practice
This repository is for learning Neural ODE and related practice. Come on!!!!!!!

## for the first step: 
1 Choose a DE system eg. $`\frac{dx}{dt}`$ = $`\lambda * x`$, and randomly select the parameters in this system. then we can have a lot of values of (t,x), which is original data of this system.

2 Define ODE system dudt=  .eg.we give $`\lambda`$ a guess value $`\lambda_{0}`$

3 Define Loss function

4 Run K epochs and then estimate the best $`\lambda`$ to make it recover the original $`\lambda`$. 

5 When solve problems above, try more complicated equation systems with more parameters!

## Step 2:
1 learn deeper in julia.(have uploaded learning diffequation.jl file)

2 use SIR model and think about the use. Have a practice to use the model. (here is practice and learning about SIR model in repository)

3 solve the questions this week and ask for advice.
4 there are questions about estimating parameter in julia(diffeqflux) and learning diffequation.jl(uploaded).

## Step 3:
1 build a chain model: Dense layer - Neural ode - Dense layer
2 use raw data mnist to test the model- it works
3 next step use data into the model

