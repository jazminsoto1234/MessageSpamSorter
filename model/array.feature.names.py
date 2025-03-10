import pandas as pd
import pickle

df = pd.read_csv("/home/jazmin/Escritorio/Projects/MessageSpamSorter/emails.csv")

df = df.drop(columns=["Email No.", "Prediction"])  
#df = pd.get_dummies(df, columns=['Email No.'], drop_first=True)

features = df.columns

#print(features[0])

pickle.dump(features, open("features_names_df.pkl", "wb"))




