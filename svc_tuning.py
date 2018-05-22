import optunity
import optunity.metrics
import helper
from sklearn.feature_extraction.text import CountVectorizer

# 1. Create the data set
# training data
strategy_instance = helper.strategy()
data = strategy_instance.class0 + strategy_instance.class1
labels = [False] * len(strategy_instance.class0) + [True] * len(strategy_instance.class1)

# test_data
# test_data = './test_data.txt'
# with open(test_data,'r') as file:
#     x_test = [line.strip().split(' ') for line in file]
# for i in range(len(x_test)):
#     x_test[i] = " ".join(x_test[i])
# vectors_test = vectorizer.fit_transform(x_test)
# y_test = [True] * len(x_test)

default_parameters = {'gamma': 'auto', 'C': 1.0, 'degree': 3, 'kernel': 'rbf', 'coef0': 0.0}
vectorizer = CountVectorizer(max_df=1.0, min_df=1, binary=True, ngram_range=(1, 1), max_features=5720, lowercase=False)


# 2. Performance of an SVC with default hyperparameters
# compute area under ROC curve of default parameters
@optunity.cross_validated(x=data, y=labels, num_folds=5)
def svm_default_auroc(x_train, y_train, x_test, y_test):
    for i in range(len(x_train)):
        x_train[i] = " ".join(x_train[i])
    x_train = vectorizer.fit_transform(x_train)
    model = strategy_instance.train_svm(default_parameters, x_train, y_train)
    for i in range(len(x_test)):
        x_test[i] = " ".join(x_test[i])
    x_test = vectorizer.transform(x_test)
    decision_values = model.decision_function(x_test)
    auc = optunity.metrics.roc_auc(y_test, decision_values)
    return auc

svm_default_auroc()

# 3. Tune SVC with RBF kernel
# we will make the cross-validation decorator once, so we can reuse it later for the other tuning task
# by reusing the decorator, we get the same folds etc.
cv_decorator = optunity.cross_validated(x=data, y=labels, num_folds=5)
#
# def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, logGamma):
#     tuned_parameters = {'gamma': 10 ** logGamma, 'C': C, 'degree': 3, 'kernel': 'rbf', 'coef0': 0.0}
#     for i in range(len(x_train)):
#         x_train[i] = " ".join(x_train[i])
#     x_train = vectorizer.fit_transform(x_train)
#     model = strategy_instance.train_svm(tuned_parameters, x_train, y_train)
#     for i in range(len(x_test)):
#         x_test[i] = " ".join(x_test[i])
#     x_test = vectorizer.transform(x_test)
#     decision_values = model.decision_function(x_test)
#     auc = optunity.metrics.roc_auc(y_test, decision_values)
#     return auc
#
# svm_rbf_tuned_auroc = cv_decorator(svm_rbf_tuned_auroc)
# # this is equivalent to the more common syntax below
# # @optunity.cross_validated(x=data, y=labels, num_folds=5)
# # def svm_rbf_tuned_auroc...
#
# svm_rbf_tuned_auroc(C=1.0, logGamma=0.0)
#
# # find optimal parameters
# optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=150, C=[0, 10], logGamma=[-5, 0])
# # when running this outside of IPython we can parallelize via optunity.pmap
# # optimal_rbf_pars, _, _ = optunity.maximize(svm_rbf_tuned_auroc, 150, C=[0, 10], gamma=[0, 0.1], pmap=optunity.pmap)
#
# print("Optimal parameters: " + str(optimal_rbf_pars))
# print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)
#
# df = optunity.call_log2dataframe(info.call_log)
# print("Top 10 parameters for RBF kernel:")
# print(df.sort_values('value', ascending=False)[:10])

# 4. Tune SVC without deciding the kernel in advance
# Define parameter search space
space = {'kernel': {'linear': {'C': [0, 32768]},
                    'rbf': {'logGamma': [-15, 3], 'C': [0, 32768]},
                    'poly': {'logGamma': [-15, 3], 'degree': [2, 10], 'C': [0, 32768], 'coef0': [0, 10]},
                    'sigmoid': {'logGamma': [-15, 3], 'C': [0, 32768], 'coef0': [0, 10]}
                    }
         }

# modify the objective function to cope with conditional hyperparameters
def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    parameters = {'gamma': 'auto', 'C': 1.0, 'degree': 3, 'kernel': 'rbf', 'coef0': 0.0}
    if kernel == 'linear':
        parameters = {'gamma': 'auto', 'C': C, 'degree': 3, 'kernel': kernel, 'coef0': 0.0}
    elif kernel == 'poly':
        parameters = {'gamma': 10 ** logGamma, 'C': C, 'degree': degree, 'kernel': kernel, 'coef0': coef0}
    elif kernel == 'rbf':
        parameters = {'gamma': 10 ** logGamma, 'C': C, 'degree': 3, 'kernel': kernel, 'coef0': 0.0}
    elif kernel == 'sigmoid':
        parameters = {'gamma': 10 ** logGamma, 'C': C, 'degree': 3, 'kernel': kernel, 'coef0': coef0}
    else:
        raise AttributeError("Unknown kernel function: %s" % kernel)
    for i in range(len(x_train)):
        x_train[i] = " ".join(x_train[i])
    x_train = vectorizer.fit_transform(x_train)
    model = strategy_instance.train_svm(parameters, x_train, y_train)
    model.fit(x_train, y_train)
    return model

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
    model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
    for i in range(len(x_test)):
        x_test[i] = " ".join(x_test[i])
    x_test = vectorizer.transform(x_test)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

# optimize kernel function and associated hyperparameters
optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)
print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)

df = optunity.call_log2dataframe(info.call_log)
sorted_df = df.sort_values('value', ascending=False)
print("Top 10 parameters for SVC:")
print(sorted_df)

filename = "tuning_results.csv"
print("Export result:", filename)
sorted_df.to_csv(filename, sep=',', encoding='utf-8', index=False)