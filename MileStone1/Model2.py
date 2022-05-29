from sklearn import linear_model
from sklearn import metrics
import numpy as np
import time

# Modeel 2
def Model2(feature_train,likesCount_train,feature_test,likesCount_test):
    
    PassiveAggressiveRegressor_start_time = time.time()
    PassiveAggressiveRegressor = linear_model.PassiveAggressiveRegressor()
    PassiveAggressiveRegressor.fit(feature_train,likesCount_train)
    PassiveAggressiveRegressor_training_time = time.time() - PassiveAggressiveRegressor_start_time
    PassiveAggressiveRegressor_prediction_time_start = time.time()
    prediction= PassiveAggressiveRegressor.predict(feature_test)
    PassiveAggressiveRegressor_prediction_time = time.time() - PassiveAggressiveRegressor_prediction_time_start
    true_value=np.asarray(likesCount_test)[0]
    predicted_value=prediction[0]
    
    
    print('Model 2 : Passive Aggressive Regression')
    print('Co-efficient of linear regression',PassiveAggressiveRegressor.coef_)
    print('Intercept of linear regression model',PassiveAggressiveRegressor.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(likesCount_test), prediction))
    print('True value in the test is : ' + str(true_value))
    print('Predicted value in the test  is : ' + str(predicted_value))

    print('Accuracy of the model is : ' + str(metrics.r2_score(np.asarray(likesCount_test), prediction)*100))
    print('Training time of the model is : ' + str(PassiveAggressiveRegressor_training_time))
    print('Prediction time of the model is : ' + str(PassiveAggressiveRegressor_prediction_time))
