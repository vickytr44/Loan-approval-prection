import pandas as pa
import numpy as np
data_train= pa.read_csv(r'C:\Users\vicky\Desktop\train_filled data.csv')

#Adding new feature
data_train.insert(8, 'Total_Income',(data_train['ApplicantIncome']+data_train['CoapplicantIncome']))

# fill blank fields 
data_train['Dependents'].fillna(data_train['Dependents'].mode()[0], inplace=True)
data_train['Loan_Amount_Term'].fillna(data_train['Loan_Amount_Term'].mode()[0], inplace=True)
data_train['Credit_History'].fillna(data_train['Credit_History'].mode()[0], inplace=True)

#Drop unwanted tables
data_train.drop(labels = 'Loan_ID', axis = 1, inplace = True)
data_train.drop(labels = 'ApplicantIncome', axis = 1, inplace = True)
data_train.drop(labels = 'CoapplicantIncome', axis = 1, inplace = True)

data_train['LoanAmount_log'] = np.log(data_train['LoanAmount'])
data_train['TotalIncome_log'] = np.log(data_train['Total_Income'])

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  #kf = KFold(data.shape[0], n_splits=5)
  kf=KFold(n_splits=5, random_state=None, shuffle=False)
  error = []
  for train, test in kf.split(data):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History',]
classification_model(model, data_train,predictor_var,outcome_var)

from sklearn.externals import joblib
joblib.dump(model, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
model = joblib.load('model.pkl')

#model_columns = list(data_train.columns)
#joblib.dump(model_columns, 'model_columns.pkl')
#print("Models columns dumped!")



