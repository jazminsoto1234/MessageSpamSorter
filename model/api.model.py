import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, jsonify

# Create a Flask app
app = Flask(__name__)

#~/
# Load the machine learning model from a pickle file
model = pickle.load(open("/home/jazmin/Escritorio/Projects/MessageSpamSorter/model.pkl", "rb"))

modelvectorizer = pickle.load(open("/home/jazmin/Escritorio/Projects/MessageSpamSorter/vectorizer_model.pkl", "rb"))

feature_name_df = pickle.load(open("/home/jazmin/Escritorio/Projects/MessageSpamSorter/features_names_df.pkl", "rb"))


@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(Message="Success")

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    json_ = [request.json['email']]

    print(json_)
    count_matrix = modelvectorizer.fit_transform(json_)
    cont_arr = count_matrix.toarray()

    # Convert JSON data into a DataFrame
    df = pd.DataFrame(data=cont_arr, columns=modelvectorizer.get_feature_names_out()) 

    #print(modelvectorizer.get_feature_names_out())
    #print(df)
    print("")
    #print(feature_name_df)

    cont_real = []
    i = 0
    for token in  feature_name_df:
        #cont_real.append(0)
        if token in modelvectorizer.get_feature_names_out():

            cont_real.append(cont_arr[0][i])
            i+=1
        else:
            cont_real.append(0)
    
    cont_real = np.array(cont_real).reshape(1, -1)
    print(cont_real)
    
    df1 = pd.DataFrame(data=cont_real, columns= feature_name_df)


    #df = [element for element in feature_name_df]

    # Use the loaded model to make predictions on the DataFrame
    prediction = model.predict(df1)

    # Return the predictions as a JSON response
    #
    result_predict = "no-spam"
    if prediction[0] == 1:
        result_predict = "spam"

    
    return jsonify({"Prediction": result_predict})


# Run the Flask app when this script is executed
if __name__ == "__main__":
    app.run(port=3000 ,debug=True)


#To STOP receiving these emails from us Just hit *REPLY* and let us know Thanks.