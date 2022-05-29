from sklearn import linear_model
from sklearn import metrics
import numpy as np
import time
        
def Model1(feature_train,likesCount_train,feature_test,likesCount_test):                                                                    
    

    BayesianRidge_start_time = time.time()
    BayesianRidge = linear_model.BayesianRidge()
    BayesianRidge.fit(feature_train,likesCount_train)
    BayesianRidge_training_time = time.time() - BayesianRidge_start_time
    prediction= BayesianRidge.predict(feature_test)
    BayesianRidge_prediction_time = time.time() - BayesianRidge_training_time
    true_value=np.asarray(likesCount_test)[0]
    predicted_value=prediction[0]
    
    print('\n\nModel1 : Bayesian Ridge Regression')
    print('Co-efficient of linear regression',BayesianRidge.coef_)
    print('Intercept of linear regression model',BayesianRidge.intercept_)
    
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(likesCount_test), prediction))
    print('True value in the test is : ' + str(true_value))
    print('Predicted value in the test  is : ' + str(predicted_value))

    print('Accuracy of the model is : ' + str(metrics.r2_score(np.asarray(likesCount_test), prediction)*100))
    print('Training time of the model is : ' + str(BayesianRidge_training_time))
    print('Prediction time of the model is : ' + str(BayesianRidge_prediction_time))
    
