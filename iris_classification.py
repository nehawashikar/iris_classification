from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

'''
This method loads the data to train on
'''
def load_data():
    iris = datasets.load_iris() #data, target, target_names, feature_names, DESCR
    #print(iris.data)
    X, y = iris.data, iris.target
    return X, y

'''
This method does 5 fold cross validation using stratified k fold and calculates the 3 scoring metrics
'''
def cross_validation(model, X, y, n_components=None):
    #use stratified k fold to reduce bias towards one class
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'roc_auc_ovr': make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovr')
    }

    #add PCA to model via Pipeline if n_components are specified 
    if n_components:
        model = Pipeline([
            ('pca', PCA(n_components=n_components)),
            ('classifier', model)
        ])
        
    #get the results from corss validation
    results = cross_validate(model, X, y, cv=cv, return_train_score=True, scoring=scoring)
    return results

'''
This method prints the necessary test and train metrics
'''
def print_metrics(results):
    #print accuracy
    print("Average Train Accuracy:", results['train_accuracy'].mean())
    print("Average Test Accuracy:", results['test_accuracy'].mean())
    #print F1-score
    print("Average Train F1-score:", results['train_f1_macro'].mean())
    print("Average F1-score (Macro):", results['test_f1_macro'].mean())
    #print ROC AUC
    print("Average Train ROC AUC (One-vs-Rest):", results['train_roc_auc_ovr'].mean())
    print("Average ROC AUC (One-vs-Rest):", results['test_roc_auc_ovr'].mean())

X, y = load_data()
# Naive Bayes
print("------------------------Naive Bayes------------------------")
results = cross_validation(GaussianNB(), X, y)
print_metrics(results)

# Support Vector Machines
print("------------------------SVM------------------------")
results = cross_validation(SVC(probability=True, random_state=0), X, y)
print_metrics(results)

# Random Forest
print("------------------------Random Forest------------------------")
results = cross_validation(RandomForestClassifier(max_depth = 10, min_samples_leaf = 5, random_state=0, n_estimators = 200), X, y, n_components=2)
print_metrics(results)

# XGBoost
print("------------------------XGBoost------------------------")
results = cross_validation(XGBClassifier(max_depth = 3, min_child_weight = 7, subsample = 0.5, eval_metric='mlogloss', random_state=0), X, y)
print_metrics(results)

#K-Nearest Neighbors 
print("------------------------KNN------------------------")
results = cross_validation(KNeighborsClassifier(n_neighbors=8), X, y)
print_metrics(results)