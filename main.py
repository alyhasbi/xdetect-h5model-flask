import numpy as np
from PIL import Image
import tensorflow as tf
import os
import h5py
from flask import Flask, request, jsonify, send_file
from google.cloud import storage
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials
import urllib.request
import datetime
from firebase_admin import db
from firebase_admin import firestore
import pytz



load_dotenv()

# Get Firebase configuration from environment variables
firebase_config = {
    "apiKey": os.getenv("API_KEY"),
    "authDomain": os.getenv("AUTH_DOMAIN"),
    "projectId": os.getenv("PROJECT_ID"),
    "storageBucket": os.getenv("STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("MESSAGING_SENDER_ID"),
    "appId": os.getenv("APP_ID")
}

# Initialize Firebase SDK
cred = credentials.Certificate('./db-config/serviceAccount.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
bucket_name = "xdetect-img-profile" 
class_mapping = ["Mass", "Nodule", "Normal", "Pneumonia", "Tuberculosis"]


def predict_image_class(image_path, model_h5_path, class_mapping):
    # Load and preprocess the input image
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize the image to match the expected input shape of the model
    image = np.array(image) / 255.0  # Normalize the image pixels between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Verify the model.h5 file exists
    if not os.path.isfile(model_h5_path):
        print("Model file does not exist:", model_h5_path)
        return None

    # Check the contents of the model.h5 file
    with h5py.File(model_h5_path, "r") as f:
        if "model_config" not in f.attrs.keys():
            print("Invalid model.h5 file:", model_h5_path)
            return None

    # Load the trained model
    model = tf.keras.models.load_model(model_h5_path)

    # Perform inference
    predictions = model.predict(image)
    print(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_mapping[predicted_class_index]

    print(predicted_class)

    return predictions, predicted_class

def run_image_classification(image_url, model_h5_path):
    # Download the image from the provided URL
    image_path = 'temp_image.jpg'
    urllib.request.urlretrieve(image_url, image_path)

    # Define the class mapping

    # Call the predict_image_class function
    predictions, predicted_class = predict_image_class(image_path, model_h5_path, class_mapping)

    # Remove the temporary image file
    os.remove(image_path)

    return predictions, predicted_class
        

def upload_file_to_bucket(bucket_name, file_name, file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob.upload_from_file(file, content_type='application/octet-stream')

    # Make the uploaded file publicly accessible
    blob.make_public()

    public_url = blob.public_url

    return public_url

@app.route('/detect-disease/<uid>', methods=['POST'])
def upload_skin_picture(uid):
    try:
        if len(request.files) == 0:
            response = jsonify({
                'status': 'Failed',
                'message': 'Tidak ada file yang ditambahkan'
            })
            response.status_code = 400
            return response

        file = next(iter(request.files.values()))
        file_name = file.filename

        public_url = upload_file_to_bucket(bucket_name, file_name, file)

        predictions, predicted_class = run_image_classification(public_url, 'modelxdetect.h5')
        prediction_result = {}
        for index, prediction in enumerate(predictions[0]):
            prediction_result[class_mapping[index]] = f"{float(prediction) * 100}%"

        # Adjust timestamp to client's time zone
        client_timezone = pytz.timezone('Asia/Jakarta')  # Replace with the appropriate time zone
        client_timestamp = datetime.datetime.now(client_timezone)

        doc_ref = db.collection('users2').document(uid)
        doc_ref.update({
            'history': firestore.ArrayUnion([{
                'type' : 'Skin Disease Detection',
                'datetime': client_timestamp,
                'predicted_class': predicted_class,
                'detection_img': public_url,
            }])
        })

        response = jsonify({
            'type' : 'Skin Disease Detection',
            'status': 'Success',
            'message': 'Deteksi penyakit berhasil',
            'detection_img': public_url,
            'class' : predicted_class,
            'predictions': prediction_result
        })
        response.status_code = 200
        return response

    except Exception as error:
        print(error) 

        response = jsonify({
            'status': 'Failed',
            'message': 'An internal server error occurred',
            'error': str(error)
        })
        response.status_code = 500
        return response

@app.route('/history/<uid>', methods=['GET'])
def get_skin_picture_history(uid):
    try:
        # Retrieve the user document
        doc_ref = db.collection('users2').document(uid)
        doc = doc_ref.get()

        if doc.exists:
            # Retrieve the history data from the document
            history = doc.to_dict().get('history', [])

            formatted_history = []
            for item in history:
                datetime_value = item['datetime']
                datetime_utc = datetime_value.replace(tzinfo=pytz.UTC)
                datetime_local = datetime_utc.astimezone(pytz.timezone('Asia/Jakarta'))
                formatted_datetime = datetime_local.strftime("%d %B, %Y at %H:%M:%S")
                item['datetime'] = formatted_datetime
                formatted_history.append(item)

            # Reverse the order of the history list
            formatted_history.reverse()

            response = jsonify({
                'status': 'Success',
                'message': 'Skin detection history berhasil didapatkan',
                'data': formatted_history
            })
            response.status_code = 200
            return response
        else:
            response = jsonify({
                'status': 'Failed',
                'message': 'User tidak ditemukan'
            })
            response.status_code = 404
            return response

    except Exception as error:
        print(error)  # Log the error for debugging purposes

        response = jsonify({
            'status': 'Failed',
            'message': 'An internal server error occurred',
            'error': str(error)
        })
        response.status_code = 500
        return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8090)


