# Predicting water quality
### Project 1 for the 'Explainable Automated Machine Learning' (LTAT.02.023) course

## Team
Dmitri Rozgonjuk <br>
Lisanna Lehes <br>
Marilin Moor <br>
Fjodor Ševtšenko <br>

## Assignment
Each team (group of three) will work on a machine learning problem from end-to-end.

### First step :white_check_mark:
- **Choose a dataset**.
- **Build and train a baseline for comparison**.
- To construct the baseline you do the following:
  - Try a set of possible machine learning algorithms (**13 algorithms**) using their **default hyperparameters** and **choose** the one with **the highest performance** for comparison.
  
### Second step :white_check_mark:
Based on the problem at hand, you 
- study the potential pipeline structure, algorithms or feature transformers at each step, hyper-parameters ranges. 
- Use `hyperOpt` with the potential search space to beat the baseline.

### Third step
- :x: Monitor the performance of you the constructed pipeline from the previous step **across different time budgets** (number of iterations) and report **the least time budget** that you are able **to outperform the baseline**.
  - <font color = 'red' >DR comment: I understand this here so that we should plot accuracies from each trial and compare it with the accuracy in the baseline. I added this plot to the HPO NB.
  </font>

### Fourth step
- :x: **Determine** whether the **difference in performance** between the constructed pipeline and the baseline **is statistically significant**.
  - <font color = 'red' >DR comment: I guess here we should use the McNemar test (comparing the predictions of baseline vs best model). Either HW1 or HW6 should help!
  </font>


