# Predicting water quality
### Project 1 for the 'Explainable Automated Machine Learning' (LTAT.02.023) course

## Team
Dmitri Rozgonjuk <br>
Lisanna Lehes <br>
Marilin Moor <br>
(Fjodor Ševtšenko) - contributed with some discussion

## Assignment
Each team (group of three) will work on a machine learning problem from end-to-end.

### First step :white_check_mark:
- **Choose a dataset**.
  - **SOLUTION**: We chose the Water Quality Prediction dataset from kaggle.
- **Build and train a baseline for comparison**. To construct the baseline you do the following: Try a set of possible machine learning algorithms (**13 algorithms**) using their **default hyperparameters** and **choose** the one with **the highest performance** for comparison.
  - **SOLUTION**: We have now computed baselines for 13 different classifiers with their default hyperparameters. We selected the five best models based on the model accuracy.
  
### Second step :white_check_mark:
Based on the problem at hand, study the potential pipeline structure, algorithms or feature transformers at each step, hyper-parameters ranges. Use `hyperOpt` with the potential search space to beat the baseline.
  - **SOLUTION**: We have defined a hyperparameter search space for the five best baseline classifiers. When running the HPO, the best results (i.e., accuracy) are presented alongside the time it took to reach this result as well as the hyperparameter set used for the best model.

### Third step :white_check_mark:
- Monitor the performance of you the constructed pipeline from the previous step **across different time budgets** (number of iterations) and report **the least time budget** that you are able **to outperform the baseline**.
  - **SOLUTION**: The performance of the constructed HPO pipeline is monitored over all trials and output is presented when either the model beats the baseline (in this case, accuracy and runtime are presented) or if the model has improved in comparison to previous iteration(s). Furthermore, we also display the relationship between hyperparameters and model performance (accuracy). Finally, in the model test stage, we present the model that beats the baseline AND has the lowest runtime until finding the best performance model.

### Fourth step :white_check_mark:
-  **Determine** whether the **difference in performance** between the constructed pipeline and the baseline **is statistically significant**.
  - **SOLUTION**: Because the initial task wanted us to test the different between the baseline model and the model with HPO, we conducted a dependent-samples t-test. The reason for this is that when we are computing the accuracy scores for data splits, the data splits are the same for the both models.

### Final step:
- **Project write-up and presentation**
  - SOLUTION: Currently merging the different parts of pipeline into one notebook. Also creating a slide deck for a presentation.

