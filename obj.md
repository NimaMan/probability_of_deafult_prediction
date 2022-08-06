# Simdi

- Finish the lgbm and xgb models 

- main thing that works: training and testing on the 13 data! 
 - Possible questions that it brings 
    - What is its performace on lower dimensional data?
    - how to transfer this into lower dimensional data?

- Eror Analysis:
    - the main postion of the error comes from the recall. 
        - Q: why do I have this much error in the recall? And how to decrease knowing that we only take 4% recall? 
         - An option maybe to feed the data into a graph! why graph you ask? Or models that take into account a dependence between all the data that we have at the moment of the prediction

- sample 50-50 data from the dataset for taining (did not improve the performance)

-  the cat model with the simplest agg model -> using the cat variables as a cont var with LabelEncoder
- use the sum of the cat features and create and train a cat model 

- ivnestigate possible different aggregations

- run the lightgbm, xgb with a variaty of different features

- SFA
    SFA suggests that some feature may be more informative when transformed. Currently investigating combinations of features
    - What does the results of SFA imply? 
    
    -  possibly come up with a distribution of Weak learners sampling from the dist of sinfle factor analysis


- Different precisons from the official pytorch 

## Categorical data
    Transform the categorical data
    - Later pay speacial attention to the B_38
### Olan
-  I have a convolutional model that works pretty good for the C13

- I know that recall is the main bottleneck in its error metric

- I have started training weak models. -> a possible path is to train so many weak models give it to a ensemble learner that inegrates the preds 

## Roadmap

- train SFA 
    - on c13 data

- evaluate errors
- later propogate the results into the rest of the dataset

- Add the predictions from other models to the features that you have. 

- Investigate ranked probabilities

# Ensemble
Will come up with a couple of ensemble models. 
    - Add the predictions of lgbm and xgb as a feature to conv




# Results

- ContCols + Cat cols transformed -> 793 (could possibly have been improved with CV and early stopping) 

- K79 + 6 best 13 overfits significantly with 799 on validation data and somehow 785 on real test data!!


# Current repeat procedure 

## FE 



## Models 


## HPO


## CV



## Ensemble


