from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tempfile import SpooledTemporaryFile

app = Flask(__name__)
model = load_model("red_yellow_green.h5", compile=False)
class_names = open("red_yellow_green.txt", "r").readlines()


class SpooledBytesIO(SpooledTemporaryFile):
    def __init__(self, initial_bytes=None, max_size=1024, mode='w+b'):
        super().__init__(max_size=max_size, mode=mode)
        if initial_bytes:
            self.write(initial_bytes)
            self.seek(0)

    @property
    def size(self):
        self.seek(0, 2)
        size = self.tell()
        self.seek(0)
        return size


def predict_image(image):
    # Преобразуем объект SpooledBytesIO в объект PIL.Image
    image_pil = Image.open(image)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image_pil = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image_pil)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_stream = image.stream
    image_bytes = image_stream.read()
    image_stream.close()

    spooled_bytes_io = SpooledBytesIO(initial_bytes=image_bytes)

    class_name, confidence_score = predict_image(spooled_bytes_io)

    return jsonify({'prediction': class_name[2:], 'confidence': confidence_score}), 200


if __name__ == '__main__':
    app.run(debug=True)
