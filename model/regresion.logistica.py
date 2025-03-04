
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv("/home/jazmin/Escritorio/Projects/MessageSpamSorter/emails.csv")


df = pd.get_dummies(df, columns=['Email No.'], drop_first=True)


print(df.head(3))


# Informacion relevante


#print("Numero de observaciones por clase")
# print(df['Prediction'].value_counts())
# print("")


#print("Porcentaje de observaciones por clase")
#print(100 * df['Prediction'].value_counts(normalize=True))


# Dividir los datos

x = df.drop(columns = 'Prediction')
y = df['Prediction']



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) 


#Crear modelo


pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(x_train, y_train)  # Error: La iteracion es demasiado grande solucion escalar

# Score

print(pipe.score(x_test, y_test))

