import pandas as pd
import pickle
import numpy as np
from flask_swagger_ui import get_swaggerui_blueprint
from flask import Flask, request, jsonify, send_from_directory
import os
# Create a Flask app
app = Flask(__name__)



rutamodel = os.path.abspath("./model.pkl")
rutamodelvector = os.path.abspath("./vectorizer_model.pkl")
rutamodelfeature = os.path.abspath("./features_names_df.pkl")


#~/
# Load the machine learning model from a pickle file
model = pickle.load(open(rutamodel, "rb"))

modelvectorizer = pickle.load(open(rutamodelvector, "rb"))

feature_name_df = pickle.load(open(rutamodelfeature, "rb"))


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
    #df = pd.DataFrame(data=cont_arr, columns=modelvectorizer.get_feature_names_out()) 

    #print(modelvectorizer.get_feature_names_out())
    #print(df)
    #print("")
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
    #print(cont_real)
    
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



@app.route('/swagger.json', methods=['GET'])
def swagger_spec():
    # Devuelve el archivo swagger.json
    return send_from_directory('static', 'swagger.json')





SWAGGER_URL = '/swagger'  
API_URL = '/swagger.json' 

#Configuraracion 
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Ejemplo API"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)



# Run the Flask app when this script is executed
if __name__ == "__main__":
    app.run(port=3000 ,debug=True)


#To STOP receiving these emails from us Just hit *REPLY* and let us know Thanks.