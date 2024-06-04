## Mean ai experiment
 **Why** - At one point I read a book which was called [wisdom of the crowds](https://en.wikipedia.org/wiki/The_Wisdom_of_Crowds) which posed the idea of how a group of people can make better decisions than a single person. Some interesting results related to this idea are
 - Measuring the weight of an ox where the group had far better accuracy than the individual guesses.
 - Using bayesian search theory to find the lost ship [SS Central America](https://en.wikipedia.org/wiki/SS_Central_America) which was lost in 1857.
 - Open source software generally being much more robust and secure

The premise was that given a **true** value for a certain weight, location, etc the **educated** guesses of multiple people will form a distribution around that true value. Taking the mean of all those values as n goes to infinity, will result in the true value. This is a very interesting concept and I wanted to see if it could be applied to AI.

### Setup
#### Version1 
Basic 5 AI autoencoder agents, 8-4-8 trained on same dataset, same functions. Train each one of them for n epochs and for evaluation taking the mean of all the outputs and compare it to the, while also measuring the mean error of each agent.
- 5 networks, 8-4-8 identical
- trained on same dataset

**Results**
- The error of the mean was smaller with about 14.9% across multiple trials compared to the individual agents.
- Quite notably, increasing the number of intermediate layers from 4 to 6, did increase the difference between mean and individual to 19.2%

**Pitfalls**
- The agents were trained on the same dataset, with the same activation functions. Therefore all their predictions might fall on a single side of the distribution around the true mean. 
- We need to test:
  - larger networks
  - networks with different architectures ( hoping for them to "think" differently)
  - network with varying different datasets

#### Version2
Identical to version1, but the networks are 8-16-4-16-8
We will denote
- average of errors ( so each network evaluated individually ) - AE
- error of averages ( so mean of all networks and take error from that ) - EA
- error of super net ( a network trained on all the data for all epochs ) - ES

Testing 5 networks, 50 epochs each, super net is trained 5*50 epochs
**Results**
- The error of the mean was smaller with 26.3% for Tanh function. 
- Importantly, not all activation functions produce these improvments. For sigmoid the difference is only 8%, compared to 26% for tanh. This indicated that the negative values of tanh might distribute the predictions more evenly around the true mean. 
- Interestingly, using a single network of sigmoid performs about the same as a single tanh network, yet when applying the averaging method, the difference is clear
- Adding a "super network" which trains for nr_network * epochs_per_network. This network is supposed to be "the expert"
- For Tanh the error of the super net was much higher than the mean of the individual networks
- For Sigmoid the error of the super net was much lower than the mean of the individual networks
- For Sigmoid, if the average nets are trained to their highest possible accuracy, the super net will 

Testing 10 networks, 500 epochs each, super net is trained 10*500 epochs
**Results**
- This time for sigmoid:  EA is 40% smaller than AE, and EA is 20% smaller than ES

### Version3
- Adding those techniques which elimate neurons and prevent weight explosion
- Testing 10-20 networks on large epochs
- Testing randomized network architectures
- error of ultra net ( a network which is trained as all others combined and has the size of all others combined ) 

#### Found out this is actually called [ensemble learning](https://arxiv.org/pdf/2104.02395) so its already studied