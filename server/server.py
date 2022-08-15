from flask import Flask, request, jsonify
import util


def main():
    app = Flask(__name__)


    @app.route("/classify_image",methods=["GET","POST"])
    #This function does the image classification based on our saved model in the artifacts directory:
    def classify_image():
        image_data = request.form['image_data']

        response = jsonify(util.classify_image(image_data))

        response.headers.add('Access-Control-Allow-Origin', '*')

        return response

    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()

    app.run(port=5000)


if __name__ == "__main__":
    main()