## Simdi

- Eror Analysis: 

- sample 50-50 data from the dataset for taining (did not improve the performance)

-  the cat model with the simplest agg model -> using the cat variables as a cont var with LabelEncoder
- use the sum of the cat features and create and train a cat model 

- ivnestigate possible different aggregations

- run the lightgbm model
 - investigate later with your own features. Also with the aggregations that you get out of the neural net

- run SFA
    - SFA with Conv 
    - SFA with LGBM 
    - What does the results of SFA imply? possibly come up with a distribution of Weak learners sampling from the dist of sinfle factor analysis

- currently running the model with the preprop from the kaglle

- Run a ensemble over the probs out of lgbm with MLP, CNN, Att

## Roadmap

- train SFA 
    - on c13 data

- evaluate errors
- later propogate the results into the rest of the dataset