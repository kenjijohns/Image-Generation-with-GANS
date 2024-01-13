from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot
from io import BytesIO
import base64
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    model = load_model('generator_model.h5')

    # Get the number of faces from the form input
    num_faces = int(request.form['num_faces'])

    latent_points = generate_latent_points(100, num_faces)
    generated_images = generate_faces(model, latent_points)
    generated_images = (generated_images + 1) / 2.0

    # Convert images to base64 for displaying in HTML
    img_list = []
    for img in generated_images:
        buffer = BytesIO()
        pyplot.imsave(buffer, img)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_list.append(img_str)

    return render_template('generated.html', img_list=img_list)

@app.route('/interpolate', methods=['POST'])
def interpolate():
    model = load_model('generator_model.h5')
    latent_points = generate_latent_points(100, 2)

    # Get the number of steps from the form input
    num_steps = int(request.form['num_steps'])

    interpolated_points = interpolate_points(latent_points[0], latent_points[1], num_steps)
    interpolated_images = generate_faces(model, interpolated_points)
    interpolated_images = (interpolated_images + 1) / 2.0

    # Convert images to base64 for displaying in HTML
    img_list = []
    for img in interpolated_images:
        buffer = BytesIO()
        pyplot.imsave(buffer, img)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_list.append(img_str)

    return render_template('interpolated.html', img_list=img_list)

@app.route('/manipulate', methods=['GET'])
def manipulate_page():
    # Render the manipulate.html template for GET requests
    return render_template('manipulate.html')

@app.route('/manipulate', methods=['POST'])
def manipulate():
    # Handle POST requests for generating faces
    model = load_model('generator_model.h5')
    num_faces = 100
    latent_points = generate_latent_points(100, num_faces)

    # Randomly select three latent vectors for manipulation
    indices = random.sample(range(num_faces), 3)
    latent_point_1, latent_point_2, latent_point_3 = latent_points[indices]

    # Perform arithmetic operation on latent vectors
    manipulated_point = latent_point_1 - latent_point_2 + latent_point_3

    # Generate images from latent points
    img_str_1 = generate_and_convert_image(model, latent_point_1)
    img_str_2 = generate_and_convert_image(model, latent_point_2)
    img_str_3 = generate_and_convert_image(model, latent_point_3)
    img_str_manipulated = generate_and_convert_image(model, manipulated_point)

    # Render the template with the list of images
    return render_template('manipulate.html', img_str_1=img_str_1, img_str_2=img_str_2,
                           img_str_3=img_str_3, img_str_manipulated=img_str_manipulated)

@app.route('/generate_new_face', methods=['POST'])
def generate_new_face():
    model = load_model('generator_model.h5')

    # Generate new random latent points for manipulation
    num_faces = 100
    latent_points = generate_latent_points(100, num_faces)

    # Randomly select three latent vectors for manipulation
    indices = random.sample(range(num_faces), 3)
    latent_point_1, latent_point_2, latent_point_3 = latent_points[indices]

    # Perform arithmetic operation on latent vectors
    manipulated_point = latent_point_1 - latent_point_2 + latent_point_3

    # Generate images from latent points
    img_str_1 = generate_and_convert_image(model, latent_point_1)
    img_str_2 = generate_and_convert_image(model, latent_point_2)
    img_str_3 = generate_and_convert_image(model, latent_point_3)
    img_str_manipulated = generate_and_convert_image(model, manipulated_point)

    # Render the template with the list of images
    return render_template('manipulate.html', img_str_1=img_str_1, img_str_2=img_str_2,
                           img_str_3=img_str_3, img_str_manipulated=img_str_manipulated)

def generate_and_convert_image(model, latent_point):
    # Generate image from latent point
    generated_image = generate_faces(model, latent_point.reshape(1, 100))
    generated_image = (generated_image + 1) / 2.0

    # Convert image to base64 for displaying in HTML
    buffer = BytesIO()
    pyplot.imsave(buffer, generated_image[0])
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_str

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def interpolate_points(p1, p2, n_steps=10):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = [(1.0 - ratio) * p1 + ratio * p2 for ratio in ratios]
    return np.asarray(vectors)

def generate_faces(model, latent_points):
    X = model.predict(latent_points)
    return X

if __name__ == '__main__':
    app.run(debug=True)
