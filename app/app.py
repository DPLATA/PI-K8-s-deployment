import pickle
import cv2
import h5py
import numpy as np
import mahotas
import os
import glob
import cv2
import warnings
import matplotlib.pyplot as plt
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import datetime
import subprocess
from sklearn.preprocessing import MinMaxScaler

fixed_size = tuple((500, 500))
bins = 8
train_path = "dataset/train"
train_labels = os.listdir(train_path)



clf = pickle.load(open("clf.pkl", "rb"))

# Function to get image's hu_moment feature
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Function to get image's haralick feature
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# Function to get image's histogram feature
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# Storing image to be classified in a variable image
#image = cv2.imread("dataset/test/IMG_0001.jpg")
#image = cv2.resize(image, fixed_size)

# Getting image's features
#fv_hu_moments = fd_hu_moments(image)
#fv_haralick   = fd_haralick(image)
#fv_histogram  = fd_histogram(image)

# Getting image's global feature which is a concatanation of three previous features and scaling image
#global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
#scaler = MinMaxScaler(feature_range=(0, 1))
#rescaled_feature = scaler.fit_transform(global_feature.reshape(-1,1))

# Getting prediction as a label and inserting label in image
#prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]
#cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

# Showing the image with its corresponding classification label label with matplotlib
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.show()





#----------------------------------------FLASK SERVER----------------------------------------------------

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]iasdfffsd/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


UPLOAD_FOLDER = 'UPLOAD_FOLDER/'
UPLOAD_FOLDER1 = '.'

def CreateNewDir():
    print("I am being called")
    print(UPLOAD_FOLDER)
    global UPLOAD_FOLDER1
    UPLOAD_FOLDER1 = UPLOAD_FOLDER+datetime.datetime.now().strftime("%d%m%y%H")
    print(UPLOAD_FOLDER1)
    cmd="mkdir -p %s && ls -lrt %s"%(UPLOAD_FOLDER1,UPLOAD_FOLDER1)
    output = subprocess.Popen([cmd], shell=True,  stdout = subprocess.PIPE).communicate()[0]

    """if "total 0" in output:
        print("Success: Created Directory",UPLOAD_FOLDER1)
    else:
        print("Failure: Failed to Create a Directory (or) Directory already Exists",UPLOAD_FOLDER)"""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            UPLOAD_FOLDER = './upload_dir/'
            CreateNewDir()
            file.save(os.path.join(UPLOAD_FOLDER1, filename))
            image = cv2.imread(os.path.join(UPLOAD_FOLDER1, filename))
            image = cv2.resize(image, fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            scaler = MinMaxScaler(feature_range=(0, 1))
            rescaled_feature = scaler.fit_transform(global_feature.reshape(-1,1))
            prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]
            predicted_image = cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploaded', methods=['GET', 'POST'])
def uploaded_file():
	return '''
	<!doctype html>
	<title>Uploaded the file</title>
	<h1> File has been Successfully Uploaded </h1>
	'''

if __name__ == '__main__':
      app.secret_key = 'super secret key'
      app.config['SESSION_TYPE'] = 'filesystem'
      sess.init_app(app)
      app.debug = True
      app.run()
