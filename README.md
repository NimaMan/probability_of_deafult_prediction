
**Probability of Deafult Prediction**

# Discussion 
The data has (N, T, F) dimensions. The T may consists of some ime related information. The intial experiments with the Conv nets gives a relatively okay result. 

# Define the steps

## Step I: Simplify the problem
In this section I will be looking for ways to simplify the problem Both from the data side and from the modelling side. 

### Data 
The fists step from the data side could be only working with the dataset of the customer with length 13. These cutomers constitute 84 and 87 percent of the data 


### Features


# Findings
- REsidual + averaging performs okay 

# todo
## Major
- Determine different thresholds for feature selection
- normalize data with the quantiles (or quntile transfomer) 
- Add the cat variables to the model
- predict the NaN data using the rest of the data
- Add the catboost, look into its implementaion. Checkout how we do it with mixed integer linear programing. 


## Details
- Try sampling equal amount of data from the labels
- Try es on the conv net with initial trained start -> smapling from a relvatively okay model starts with a worse
    - try the cmaes itself (did not perform the best with a big batch size and not init params)
    - paly around with the sigma_init (submitted a bcmaes with 5)
- 
- add get name to the models itself
- experiment with different sets of columns (for now just focus on the cont)
- try conv net with shuffeling (did not affect the traning procedure much)
- normalize data by 98 percentile
- experiment with pooling and attention (initiall trials with maxpoolin and multihead attention did not workout well) 
- give numbers to the features for easier feature manipoulatioin

# Simplify
identify possible paths to a solution.
 - List the possible causes of the problem

## Assumptions
- Nan values are replaced with 0

## 