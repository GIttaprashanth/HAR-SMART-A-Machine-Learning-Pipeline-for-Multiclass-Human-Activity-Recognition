#######################################
# Importing required librarie Pipeline#
#######################################



print("Step 1: Required librarie imported successfully")
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC



####################
# To ignore warning#
####################

import warnings
warnings.filterwarnings("ignore")



###################################
#        Loading the dataset      #
###################################
print("Step 2: Created DataFrame successfully")

data = pd.read_csv(r"C:\Users\gittabunny\Desktop\project\train.csv")
data_set = pd.read_csv(r"C:\Users\gittabunny\Desktop\project\test.csv")



######################################
#   checking shape of the dataset    #
######################################
print("number of rows:",data.shape[0])
print("number of columns:",data.shape[1])


#######################################
#   checking for duplicate values     #
#######################################


print("Step 3:Handling duplicate values completed successfully ")

dup = data.duplicated().any() # For Rows

duplicated_columns = data.columns[data.T.duplicated().tolist()] # For columns
len(duplicated_columns) # Length of duplicate columns

data = data.drop(duplicated_columns,axis = 1) # duplicated columns(21) has been droped
data.shape 

######################################
#    checking for missing values     #
###################################### 


print("Step:4 Taking care of  missing values")

data.isnull().sum

data_group = data.groupby('Activity').size()
print(data_group)

##########################################
# preparing features as x and target as y#
##########################################

print("Step:5 preparing features done successfully")

x = data.drop(['Activity'],axis = 1)
y = data['Activity']

#############################################
# applying LabelEnocding on Activity column #
#############################################

print("Step:6 successfully applyed LabelEnocding on Activity column") 
le = LabelEncoder()
y = le.fit_transform(y)



########################################################
# preparing dataset into training set and tessting set #
########################################################

print("step:7 spliting data x_train,x_test&y_train,y_test")

x_train,x_test,y_train,y_test = train_test_split(
x,
y,
test_size = 0.3,
random_state = 0
)


#################
# Model Pipeline#
#################

print("Step 8: model_pipeline fcuntion created done successfully")

def model_pipeline(model):
    pipeline = IMBPipeline(steps=[
        ('scaler', MinMaxScaler()),  # Scale all numeric features
        ('smote', SMOTE(random_state=42)),
        ('feature_selection', SelectKBest(score_func=chi2, k=20)),  # Select top 20 features
        ('model', model)
    ])
    return pipeline

###################################
# Function to evaluate classifiers#
###################################
def select_model(X, y):
    classifiers = {
        "RandomForest": RandomForestClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "BernoulliNB": BernoulliNB(),
        "SVC": SVC(probability=True)  # Enable predict_proba for AUC
    }

    results = pd.DataFrame(columns=['model', 'run_time', 'roc_auc'])

    for name, clf in classifiers.items():
        print(f"\nTraining with {name}")
        start = time.time()

        pipe = model_pipeline(clf)

        try:
            cv_score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc_ovr')  # Multiclass AUC
            elapsed = round((time.time() - start) / 60, 2)

            results = pd.concat([results, pd.DataFrame([{
                'model': name,
                'run_time': elapsed,
                'roc_auc': cv_score.mean()
            }])], ignore_index=True)

        except Exception as e:
            print(f"Model {name} failed: {e}")

    return results.sort_values(by='roc_auc', ascending=False)

###################################
# Run model selection and display #
###################################

print("\nStep 9: Evaluating models with cross-validation")
model_results = select_model(x_train, y_train)

print("\nStep 10: Model comparison table")
print(model_results)


################################
# Train the best model again   #
################################

print("Step 11: Retraining the best model on full training set")

best_model_name = model_results.iloc[0]['model']
best_model = {
    "RandomForest": RandomForestClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "BernoulliNB": BernoulliNB(),
    "SVC": SVC(probability=True)
}[best_model_name]

pipeline = model_pipeline(best_model)
pipeline.fit(x_train, y_train)

######################
#  Make predictions  #
######################

print("\nStep 12: Predicting on test set")
y_pred = pipeline.predict(x_test)


###############################
# Step 13: Evaluation Scores  #
###############################

print("Step 13: Model Evaluation on Test Set")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Use try-except in case ROC AUC fails due to shape or scoring issue
try:
    print("ROC AUC:", roc_auc_score(y_test, pipeline.predict_proba(x_test), multi_class='ovr'))
except:
    print("ROC AUC could not be computed for this classifier.")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
  