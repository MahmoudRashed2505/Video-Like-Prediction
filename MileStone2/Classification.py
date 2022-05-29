import numpy as np
from numpy.random.mtrand import logistic # Used for handling numbers
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt

def Classification(X_train,Y_train,X_test,Y_test,hypertune=False):
    
    
    
    #logistic regression
    lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear') #one vs all
    start=time.time()
    lm.fit(X_train, Y_train)
    stop=time.time()
    LogisticRegressionTrainingTime=stop-start
    start=time.time()
    Logisticprediction = lm.predict(X_test)
    stop=time.time()
    LogisticRegressionTestingTime=stop-start
    Logisticaccuracy = np.mean(Logisticprediction == Y_test)
    
    print("\033[1;32;40m"+"Model 1: Logistic Regression"+"\033[1;32;40m")
    print("Logistic Accuracy: ",Logisticaccuracy*100)
    print("Logistic Regression MSE: ",mean_squared_error(Y_test, Logisticprediction))
    print("Logistic Regression R2 Score: ",r2_score(Y_test, Logisticprediction))

    print("Logistic Regression Training Time: ",str(LogisticRegressionTrainingTime))
    print("Logistic Regression Prediction Time: ",str(LogisticRegressionTestingTime))
    
    parameters = {
        'penalty' : ['l1','l2'],
        'C'       : np.logspace(-3,3,7),
        'solver'  : ['liblinear'],
    }

    logreg = lm
    LogisticRegressionGridSearch = GridSearchCV(logreg,
                    param_grid = parameters,
                    scoring='accuracy',
                    cv=10)

    if hypertune:
        LogisticRegressionGridSearch.fit(X_train,Y_train)
        print("\n\n"+"\033[1;32;40m"+"Hyperparameter Tuning..."+"\033[1;32;40m")
        print("Tuned Hyperparameters :", LogisticRegressionGridSearch.best_params_)
        print("Accuracy :",LogisticRegressionGridSearch.best_score_*100)
    



    #DT
    DT = tree.DecisionTreeClassifier(max_depth=3)
    start=time.time()
    DT.fit(X_train,Y_train)
    
    stop=time.time()
    DTtrainingTime=stop-start
    start=time.time()
    DTprediction = DT.predict(X_test)
    stop=time.time()
    DTtestingTime=stop-start
    DTaccuracy = np.mean(DTprediction == Y_test)
    
    print("\n\n"+"\033[1;32;40m"+"Model 2: Decesion Tree"+"\033[1;32;40m")
    print("DT MSE: ",mean_squared_error(Y_test, DTprediction))
    print("DT R2 Score: ",r2_score(Y_test, DTprediction))
    print("DT Accuracy: ",DTaccuracy*100)

    print("DT Training Time: ",str(DTtrainingTime))
    print("DT Testing Time: ",str(DTtestingTime))

    tree_para = {'criterion':['gini','entropy'],'max_depth':[3,4,5]}
    DTGridSearch = GridSearchCV(DT, tree_para,scoring='accuracy',
                    cv=10)
    
    if hypertune:
        DTGridSearch.fit(X_train,Y_train)
        print("\n\n"+"\033[1;32;40m"+"Hyperparameter Tuning..."+"\033[1;32;40m")
        print("Tuned Hyperparameters :", DTGridSearch.best_params_)
        print("Accuracy :",DTGridSearch.best_score_*100)


    
    print("\n\n"+"\033[1;32;40m"+"Model 3: SVM"+"\033[1;32;40m")
    print("Loading......")
    #svm
    svc = svm.SVC(kernel='linear',C=10)
    start = time.time()
    svc.fit(X_train, Y_train)
    stop = time.time()
    SVMTrainingTime = stop-start
    SVMprediction = svc.predict(X_test)
    start = time.time()
    SVMprediction = svc.predict(X_test)
    stop = time.time()
    SVMTestingTime = stop-start
    SVMaccuracy = np.mean(SVMprediction == Y_test)

    print("\n\n"+"\033[1;32;40m"+"Model 3: SVM"+"\033[1;32;40m")
    print("SVM MSE: ",mean_squared_error(Y_test, SVMprediction))
    print("SVM R2 Score: ",r2_score(Y_test, SVMprediction))
    print("SVM Accuracy: ",SVMaccuracy*100)

    print("SVM Training Time: ",str(SVMTrainingTime))
    print("SVM Prediction Time: ",str(SVMTestingTime))


    SVMparameters = {'C': [0.1, 1, 10],
                'kernel': ["linear", "poly"]}

    SVMGridSearch = GridSearchCV(svc, SVMparameters, scoring='accuracy',
                    cv=10)
    
    if hypertune:
        SVMGridSearch.fit(X_train, Y_train)
        print("\n\n"+"\033[1;32;40m"+"Hyperparameter Tuning..."+"\033[1;32;40m")
        print("Tuned Hyperparameters :", SVMGridSearch.best_params_)
        print("Accuracy :",SVMGridSearch.best_score_*100)



    
    X = ['Logistic Regression', 'Decision Tree', 'SVM']
    trainingTime = [LogisticRegressionTrainingTime, DTtrainingTime, SVMTrainingTime]
    testingTime = [LogisticRegressionTestingTime, DTtestingTime, SVMTestingTime]
    acc= [Logisticaccuracy,DTaccuracy,SVMaccuracy]

    X_axis = np.arange(len(X))

    plt.ylim(0,1)
    plt.bar(X_axis , trainingTime, 0.2, label='TrainingTime')
    plt.bar(X_axis + 0.2, testingTime, 0.2, label='TestingTime')
    plt.bar(X_axis + 0.4, acc, 0.2, label='Accuracy')

    plt.xticks(X_axis, X)
    #plt.title("Number of Students in each group")
    plt.legend()
    plt.show()


