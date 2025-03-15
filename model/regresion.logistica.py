
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle  # Para transformar el script a un modelo usable en apis

import os
rutacsv = os.path.abspath("./emails.csv")



df = pd.read_csv(rutacsv)


#df = pd.get_dummies(df, columns=['Email No.'], drop_first=True)

df = df.drop(columns=["Email No."]) 

#print(df.head(3))


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

pipe.predict(x_test)

# Save the trained model to a file named "model.pkl"
pickle.dump(pipe, open("model.pkl", "wb"))



#print(pipe.score(x_test, y_test))


#obtenemos las columnas

#col_names = df[1:1]

#print(col_names)

#def vectorizar(data):
    

# Solucion en base a las columnas, bien vectorizar el texto en python  con sus frecuencias y el node pasa a python 

#from sklearn.feature_extraction.text import TfidfVectorizer

#tfidf_vectorizer = TfidfVectorizer(analyzer=tokenize_phrase)

#tfidf_vectorizer.fit(corpus)
#tfidf_vectors = tfidf_vectorizer.transform(corpus)

