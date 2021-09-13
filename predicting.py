import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


heartDisease_df = pd.read_csv("heart_cleveland_upload.csv") # there are 297 rows and 14 columns in the dataset

x = heartDisease_df.iloc[:, 0:13]
y = heartDisease_df.iloc[:, 13]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3) # x training set has 178 rows and 13 columns

# Chi2 is for categorical values and AnovaF is for numerical values
chi2BestFeatures = SelectKBest(score_func=chi2, k=10).fit(x, y)
scoresChi2 = pd.DataFrame({"feature": x.columns, "score": chi2BestFeatures.scores_})
scoresChi2 = scoresChi2.sort_values(by="score")
print(scoresChi2)

anovaFBestFeatures = SelectKBest(score_func=f_classif, k=10).fit(x, y)
scoresAnovaF = pd.DataFrame({"feature": x.columns, "score": anovaFBestFeatures.scores_})
scoresAnovaF = scoresAnovaF.sort_values(by="score")
print(scoresAnovaF)

# Dropping the columns for fasting blood sugar, resting ECG results, and resting blood pressure after
# looking at the graphs and the scores from SelectKBest
x_bestTenFeatures_train = x_train.drop(["fbs", "restecg", "trestbps"], axis=1)
x_bestTenFeatures_test = x_test.drop(["fbs", "restecg", "trestbps"], axis=1)

# Standardizing the data
x_bestTenFeatures_train_scaled = MinMaxScaler().fit_transform(x_bestTenFeatures_train)
x_bestTenFeatures_test_scaled = MinMaxScaler().fit_transform(x_bestTenFeatures_test)

"""
# This is to separate into a cross validation set and a test set, so the resulting sets would be 60% training set,
# 20% cross validation set, and 20% test set
"""

"""
Logistic Regression Model
"""
logRegModel = LogisticRegression().fit(x_bestTenFeatures_train_scaled, y_train)
y_train_pred = logRegModel.predict(x_bestTenFeatures_train_scaled)
print(logRegModel.score(x_bestTenFeatures_test_scaled, y_test))

# Creating learning curve for logistic regression to see if there's any bias/variance in training model
train_sizes, logRegTrain_scores, logRegCV_scores = learning_curve(logRegModel, X=x_bestTenFeatures_train_scaled,
                                                                  y=y_train, cv=5)

logRegTrain_scores_mean = np.mean(logRegTrain_scores, axis=1)
logRegCV_scores_mean = np.mean(logRegCV_scores, axis=1)

learningCurveFig = plt.figure(figsize=(10, 8))
ax = learningCurveFig.add_subplot()

ax.plot(train_sizes, logRegTrain_scores_mean, label="Training Score", color="b")
ax.plot(train_sizes, logRegCV_scores_mean, label="Cross-Validation Score", color="orange")

ax.set_title("Learning Curve for a Logistic Regression Model")
ax.set_yticks(np.arange(0, 1.01, 0.1))

ax.set_xlabel("Training Examples Size")
ax.set_ylabel("Accuracy Score")

ax.legend()
plt.show()

"""
The learning curve shows that the model may benefit from having more data, but that wouldn't make a huge difference.
The model doesn't show any signs of high bias or variance.
"""

# Creating validation curve for logistic regression
params = np.logspace(-3, 1, 5)
logRegTrain_validation_scores, logRegTest_validation_scores = validation_curve(logRegModel, X=x_bestTenFeatures_train_scaled,
                                                                 y=y_train, cv=5, param_name="C", param_range=params)

logRegTrain_validation_scores_mean = np.mean(logRegTrain_validation_scores, axis=1)
logRegTest_validation_scores_mean = np.mean(logRegTest_validation_scores, axis=1)

validationCurveFig = plt.figure(figsize=(10, 8))
ax1 = validationCurveFig.add_subplot()

ax1.plot(params, logRegTrain_validation_scores_mean, label="Training Score", color="b")
ax1.plot(params, logRegTest_validation_scores_mean, label="Test Score", color="orange")

ax1.set_title("Validation Curve for a Logistic Regression Model")
ax1.set_yticks(np.arange(0, 1.01, 0.1))

ax1.set_xlabel("Regularization Parameter C")
ax1.set_ylabel("Accuracy Score")

ax1.legend()
plt.show()

"""
The validation plot shows that changing the regularization parameter C makes very little difference in accuracy 
after a certain point, so the C = 1 in the logistic regression model doesn't need to be changed.
"""

# Getting the final accuracy value for logistic regression model by averaging the accuracy scores
# from doing K-Folds Cross Validation
x_bestTenFeatures_full = x.drop(["fbs", "restecg", "trestbps"], axis=1)
kf_cv_logReg = KFold(n_splits=5, shuffle=True, random_state=3)
kf_accuracy_logReg = []

for trainIndices, testIndices in kf_cv_logReg.split(x_bestTenFeatures_full):
    x_train_logReg_kf, x_test_logReg_kf = x_bestTenFeatures_full.iloc[trainIndices], x_bestTenFeatures_full.iloc[testIndices]
    y_train_logReg_kf, y_test_logReg_kf = y.iloc[trainIndices], y.iloc[testIndices]

    x_train_logReg_kf_scaled = MinMaxScaler().fit_transform(x_train_logReg_kf)
    x_test_logReg_kf_scaled = MinMaxScaler().fit_transform(x_test_logReg_kf)

    logRegModel_kf = LogisticRegression().fit(x_train_logReg_kf_scaled, y_train_logReg_kf)
    kf_accuracy_logReg.append(logRegModel_kf.score(x_test_logReg_kf_scaled, y_test_logReg_kf))

finalAccuracy_logRegModel = np.mean(kf_accuracy_logReg)
print("Final Accuracy (Logistic Regression): ", finalAccuracy_logRegModel) # Final accuracy is 84.5 %


"""
Decision Tree Model
"""
x_bestFeatures_train_tree = x_bestTenFeatures_train.drop(["age"], axis=1)
x_bestFeatures_test_tree = x_bestTenFeatures_test.drop(["age"], axis=1)
x_bestFeatures_full_tree = x_bestTenFeatures_full.drop(["age"], axis=1)

tree = DecisionTreeClassifier(criterion="gini", splitter="best")
tree.fit(x_bestFeatures_train_tree, y_train)

# Creating learning curve for the original decision tree to see if there's any bias/variance in training model
decTree_train_sizes, decTreeTrain_scores, decTreeCV_scores = learning_curve(tree, X=x_bestFeatures_train_tree,
                                                                            y=y_train, cv=5)

decTreeTrain_scores_mean = np.mean(decTreeTrain_scores, axis=1)
decTreeCV_scores_mean = np.mean(decTreeCV_scores, axis=1)

decTree_learningCurveFig = plt.figure(figsize=(10, 8))
ax2 = decTree_learningCurveFig.add_subplot()

ax2.plot(decTree_train_sizes, decTreeTrain_scores_mean, label="Training Score", color="b")
ax2.plot(decTree_train_sizes, decTreeCV_scores_mean, label="Cross-Validation Score", color="orange")

ax2.set_title("Learning Curve for the Original Decision Tree Model")
ax2.set_yticks(np.arange(0, 1.01, 0.1))

ax2.set_xlabel("Training Examples Size")
ax2.set_ylabel("Accuracy Score")

ax2.legend()
plt.show()

# Using GridSearchCV to find the best parameters for a decision tree model
gridParams_decTree = {"max_depth": [3, 4, 5, 6, 7], "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      "max_leaf_nodes": [ 40, 45, 50, 55, 60]}
gridSearch_decTree = GridSearchCV(tree, param_grid=gridParams_decTree, scoring="f1", cv=5)

gridSearch_decTree.fit(x_bestFeatures_train_tree, y_train)
depth, leaf_samples, max_leaves = gridSearch_decTree.best_params_.values()

# Consider removing the following section for final version:
print("Best Decision Tree Parameters: ", gridSearch_decTree.best_params_)
print("Best Decision Tree Mean Cross-Validation Score: ", gridSearch_decTree.best_score_)

new_tree = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=depth,
                                  max_leaf_nodes=max_leaves, min_samples_leaf=leaf_samples)
new_tree.fit(x_bestFeatures_train_tree, y_train)

# Creating learning curve for the decision tree after doing GridSearchCV
decTreeParam_train_sizes, decTreeParamTrain_scores, decTreeParamCV_scores = learning_curve(new_tree, X=x_bestFeatures_train_tree,
                                                                            y=y_train, cv=5)

decTreeParamTrain_scores_mean = np.mean(decTreeParamTrain_scores, axis=1)
decTreeParamCV_scores_mean = np.mean(decTreeParamCV_scores, axis=1)

decTreeParam_learningCurveFig = plt.figure(figsize=(10, 8))
ax3 = decTreeParam_learningCurveFig.add_subplot()

ax3.plot(decTreeParam_train_sizes, decTreeParamTrain_scores_mean, label="Training Score", color="b")
ax3.plot(decTreeParam_train_sizes, decTreeParamCV_scores_mean, label="Cross-Validation Score", color="orange")

ax3.set_title("Learning Curve for a Decision Tree Model After Doing GridSearchCV")
ax3.set_yticks(np.arange(0, 1.01, 0.1))

ax3.set_xlabel("Training Examples Size")
ax3.set_ylabel("Accuracy Score")

ax3.legend()
plt.show()

"""
The original learning curve without the GridSearchCV parameters shows that the tree is overfitting due to the huge gap
between the training score mean and the cross validation score mean. It also doesn't seem like adding more datapoints will
help with the variance. 
The learning curve with the GridSearchCV parameters shows the training score mean and the cross validation score mean
converging, so this model has a better fit than the original one.  
"""

# Making a display of the decision tree in a png file
dot_file = export_graphviz(tree, feature_names=x_bestFeatures_full_tree.columns, filled=True, rounded=True,
                           class_names=["0", "1"])
graph = graphviz.Source(dot_file)
graph.render(filename="Decision Tree", format="png", cleanup=True)

# Performing K-Fold cross validation to find the final accuracy score for the new decision tree model
kf_cv_decTree = KFold(n_splits=5, shuffle=True, random_state=3)
kf_accuracy_decTree = []

for trainIndices, testIndices in kf_cv_decTree.split(x_bestFeatures_full_tree):
    x_train_decTree_kf, x_test_decTree_kf = x_bestFeatures_full_tree.iloc[trainIndices], x_bestFeatures_full_tree.iloc[testIndices]
    y_train_decTree_kf, y_test_decTree_kf = y.iloc[trainIndices], y.iloc[testIndices]

    decTreeModel_kf = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=depth,
                                             max_leaf_nodes=max_leaves, min_samples_leaf=leaf_samples)
    decTreeModel_kf.fit(x_train_decTree_kf, y_train_decTree_kf)
    kf_accuracy_decTree.append(decTreeModel_kf.score(x_test_decTree_kf, y_test_decTree_kf))

finalAccuracy_decTreeModel = np.mean(kf_accuracy_decTree)
print("Final Accuracy (Decision Tree): ", finalAccuracy_decTreeModel) # Final accuracy is 73.4 %


"""
Random Forest Model
"""

forest = RandomForestClassifier(max_features="sqrt", random_state=3)
forest.fit(x_bestFeatures_train_tree, y_train)

# Performing GridSearcHCV to find the best selected parameters for Random Forest Model
list_estimators = [85, 90, 95, 100, 105, 110, 115, 120]
gridParams_randForest = {"n_estimators": list_estimators, "max_depth": [7, 8, 9], "max_leaf_nodes": [40, 45, 50, 55, 60]}
gridSearch_randForest = GridSearchCV(forest, param_grid=gridParams_randForest, scoring="f1", cv=5)

gridSearch_randForest.fit(x_bestFeatures_train_tree, y_train)
depth_forest, max_leaves_forest, estimators = gridSearch_randForest.best_params_.values()

print("Best Parameters for Random Forest: ", gridSearch_randForest.best_params_)
print("Best Random Forest Cross-Validation Score: ", gridSearch_randForest.best_score_)

new_forest = RandomForestClassifier(n_estimators=estimators, max_features="sqrt", max_depth=depth_forest,
                                    min_samples_leaf=max_leaves_forest, random_state=3)
new_forest.fit(x_bestFeatures_train_tree, y_train)

# Creating learning curve for the original random forest model
randForest_trainSizes, randForest_trainScores, randForest_CVScores = learning_curve(forest, X=x_bestFeatures_train_tree,
                                                                                    y=y_train, cv=5)

randForest_trainScores_mean = np.mean(randForest_trainScores, axis=1)
randForest_CVScores_mean = np.mean(randForest_CVScores, axis=1)

randForest_learningCurveFig = plt.figure(figsize=(10, 8))
ax4 = randForest_learningCurveFig.add_subplot()

ax4.plot(randForest_trainSizes, randForest_trainScores_mean, label="Training Score", color="blue")
ax4.plot(randForest_trainSizes, randForest_CVScores_mean, label="Cross-Validation Score", color="orange")

ax4.set_xlabel("Number of Training Examples")
ax4.set_ylabel("Accuracy Score")

ax4.set_yticks(np.arange(0, 1.01, 0.1))
ax4.set_title("Learning Curve for Original Random Forest Model")

ax4.legend()
plt.show()

# Creating learning curve for the random forest model after doing GridSearchCV
randForestParams_trainSizes, randForestParams_trainScores, randForestParams_CVScores = learning_curve(new_forest, X=x_bestFeatures_train_tree,
                                                                                                      y=y_train, cv=5)

randForestParams_trainScores_mean = np.mean(randForestParams_trainScores, axis=1)
randForestParams_CVScores_mean = np.mean(randForestParams_CVScores, axis=1)

randForestParams_learningCurveFig = plt.figure(figsize=(10, 8))
ax5 = randForestParams_learningCurveFig.add_subplot()

ax5.plot(randForestParams_trainSizes, randForestParams_trainScores_mean, label="Training Score", color="blue")
ax5.plot(randForestParams_trainSizes, randForestParams_CVScores_mean, label="Cross-Validation Score", color="orange")

ax5.set_xlabel("Number of Training Examples")
ax5.set_ylabel("Accuracy Score")

ax5.set_yticks(np.arange(0, 1.01, 0.1))
ax5.set_title("Learning Curve for a Random Forest Model After Doing GridSearchCV")

ax5.legend()
plt.show()

"""
The original learning curve shows that there is a huge gap between the training and cross-validation scores, 
meaning that the original random forest model is overfitting. 
The new random forest model after doing GridSearchCV is a much better fit, with a small gap difference between
the training and cross-validation accuracy scores by about 0.03 or 3 %.
"""

# Performing K-Fold cross validation to get the final accuracy score for the new random forest model
kf_cv_randForest = KFold(n_splits=5, shuffle=True, random_state=3)
kf_accuracy_randForest = []

for trainIndices, testIndices in kf_cv_randForest.split(x_bestFeatures_full_tree):
    x_train_randForest_kf, x_test_randForest_kf = x_bestFeatures_full_tree.iloc[trainIndices], x_bestFeatures_full_tree.iloc[testIndices]
    y_train_randForest_kf, y_test_randForest_kf = y.iloc[trainIndices], y.iloc[testIndices]

    randForestModel_kf = RandomForestClassifier(n_estimators=estimators, max_features="sqrt", max_depth=depth_forest,
                                                max_leaf_nodes=max_leaves_forest)
    randForestModel_kf.fit(x_train_randForest_kf, y_train_randForest_kf)
    kf_accuracy_randForest.append(randForestModel_kf.score(x_test_randForest_kf, y_test_randForest_kf))

finalAccuracy_randForestModel = np.mean(kf_accuracy_randForest)
print("Final Accuracy (Random Forest): ", finalAccuracy_randForestModel) # Final accuracy is 81.8 %



