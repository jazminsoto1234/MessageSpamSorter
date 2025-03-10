import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle 

cv = CountVectorizer()

# Save the trained model to a file named "model.pkl"
pickle.dump(cv, open("vectorizer_model.pkl", "wb"))

