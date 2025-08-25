import joblib
import pandas as pd

def predict_species(model_name,sl,sw,pl,pw):

    if(model_name == 'decision_tree'):
        model = joblib.load('Iris_DecisionTreeModel.pkl')
    else:
        model = joblib.load('Iris_KNN.pkl')
    
    
    sample_df = pd.DataFrame([[sl, sw, pl, pw]],columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


    y_pred = model.predict(sample_df)


    iris_target = ['setosa' , 'versicolor' , 'virginica'] 
    return iris_target[y_pred[0]]



