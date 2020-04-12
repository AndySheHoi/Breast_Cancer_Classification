import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# import the dataset
cancer = load_breast_cancer()

# np.c_[cancer['data'], cancer['target']], append 'target' to 'data' as a column
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# =============================================================================
# import matplotlib.pyplot as plt
# # check the correlation between the variables 
# # Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
# plt.figure(figsize=(20,10)) 
# sns.heatmap(df_cancer.corr(), annot=True) 
# 
# # plot the correlations between different variables
# sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
# 
# # plot one of the correlations
# sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
# 
# # count the sample
# sns.countplot(df_cancer['target'], label = "Count")
# =============================================================================


# split the dataset
X = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fit a Kernel SVM to the training set, default kernel = rbf
svc = SVC()
svc.fit(X_train, y_train)

# predict the test set result
y_predict = svc.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_predict)

# visualize the test set result
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))

# K-fold cross validation
acc = cross_val_score(svc, X_train, y_train, cv = 10)
mean_acc = acc.mean()  # 0.976

# set parameters for GridSearchCV()
# different combinations of C and gamma will lead to various results

# default C = 1.0
c_param = []
for i in range(101):
    c_param.append(0.1 + i*0.1)
    
# default gamma = 1 / (n_features * X.var()) = 0.0353    
gamma_param = []
for i in range(200):
    gamma_param.append(0.01 + i*0.002)
    
param = {'C': c_param, 'gamma': gamma_param, 'kernel': ['rbf']}

# grid search with cross valiidation
# n_jobs = -1 means using all processors
gs = GridSearchCV(svc, param, scoring = 'accuracy', cv = 10, n_jobs = -1)
gs = gs.fit(X_train, y_train)

best_acc = gs.best_score_  # 0.985
best_param = gs.best_params_
gs_predict = gs.predict(X_test)

# visualize the grid search result
cm_gs = confusion_matrix(y_test, gs_predict)
sns.heatmap(cm_gs, annot=True)
print(classification_report(y_test, gs_predict))
