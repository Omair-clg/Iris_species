# %%
import joblib
import pandas as pd
model = joblib.load('Iris_DecisionTreeModel.pkl')
val1 = float(input("Sepal Length in cm: "))
val2 = float(input("Sepal Width in cm: "))
val3 = float(input("Petal Length in cm: "))
val4 = float(input("Petal Width in cm: "))

sample_df = pd.DataFrame([[val1, val2, val3, val4]],columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


y_pred = model.predict(sample_df)


iris_target = ['setosa' , 'veriscolor' , 'virginica'] 
print('\n\nPredicted Iris Species: '+ iris_target[y_pred[0]]+'\n\n')

input("Press Enter to close the program...")



