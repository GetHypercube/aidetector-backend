from flask import Flask, request, jsonify
from dmdetector import process_image as dm_process_image
from gandetector import process_image as gan_process_image
import json

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a temporary location
    file_path = '/tmp/' + file.filename
    file.save(file_path)

    # Run DM Detector
    dm_results = dm_process_image(file_path, debug=False)

    # Run GAN Detector
    gan_results = gan_process_image(file_path, debug=False)

    # Combine results
    results = {
        'DM_Detector_Results': dm_results,
        'GAN_Detector_Results': gan_results
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
