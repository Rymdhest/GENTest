from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from scipy.linalg import sqrtm

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sqrtm', methods=['POST'])
def calculate_sqrtm():
    data = request.json
    matrix = np.array(data['data'])
    try:
        print(f"incoming sqrtm matrix of size: {matrix.shape}")
        result_matrix = sqrtm(matrix)
        if np.iscomplexobj(result_matrix):
            result_matrix = result_matrix.real
        print("Done calculating")
        return jsonify({'result': result_matrix.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/covariance', methods=['POST'])
def calculate_covariance():
    data = request.json
    matrix = np.array(data['data'])
    try:
        print(f"incoming covariance matrix of size: {matrix.shape}")
        result_matrix = np.cov(matrix)
        print("Done calculating")
        return jsonify({'result': result_matrix.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
