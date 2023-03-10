# my-lair-a > app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import tensorflow as tf
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the Inception V3 model
model = tf.keras.applications.InceptionV3(weights='imagenet')


# Define the function to extract features from the video frames
def extract_features(frame):
    # Resize the frame to the input size of the model
    img = cv2.resize(frame, (299, 299))
    # Preprocess the image
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    # Extract the features using the model
    features = model.predict(np.expand_dims(img, axis=0))[0]
    return features


# Upload files

UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
console_text = []

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
        os.mkdir('uploads')
    except FileExistsError:
        print("folder exists")
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            if file.content_length <= MAX_FILE_SIZE:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File uploaded successfully')
                search_objects(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('uploaded_file', filename=filename))
            else:
                flash(f'File size exceeds the limit of {MAX_FILE_SIZE / (1024 * 1024)} MB')
        else:
            flash('Invalid file type')
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('result.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4'}


# END UPLOAD FILES

# Define the function to search for objects in the video
def search_objects(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the frames per second of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Initialize an empty list to store the features of each frame
    features_list = []
    # Read the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Extract the features from the frame
        features = extract_features(frame)
        # Append the features to the list
        features_list.append(features)
    # Release the video file
    cap.release()
    # Convert the features list to a numpy array
    features_array = np.array(features_list)
    # Compute the mean of the features array
    mean_features = np.mean(features_array, axis=0)
    # Print the top 5 predicted labels and their probabilities
    pred_probs = tf.keras.applications.inception_v3.decode_predictions(np.expand_dims(mean_features, axis=0), top=5)[0]

    for pred_prob in pred_probs:
        which = str(str(pred_prob[1]) + ':' + str(pred_prob[2]))
        console_text.append(which)
        flash(which)
        print(pred_prob[1], ':', pred_prob[2])


# Creating a route to a template is done by:
# @app.route('/')
# def index():
#     return render_template('index.html')

# Returning an API response can be done like:
@app.route('/response', methods=['GET', 'POST'])
def response():
    return {"Message": "Success"}


if __name__ == '__main__':
    app.run()
