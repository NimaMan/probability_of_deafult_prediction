## Simdi

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

- run the lightgbm model
 - investigate later with your own features. Also with the aggregations that you get out of the neural net

- run SFA
    - SFA with Conv 
    - SFA with LGBM 
    - What does the results of SFA imply? possibly come up with a distribution of Weak learners sampling from the dist of sinfle factor analysis

- currently running the model with the preprop from the kaglle

- Run a ensemble over the probs out of lgbm with MLP, CNN, Att

- Different precisons from the official pytorch 

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