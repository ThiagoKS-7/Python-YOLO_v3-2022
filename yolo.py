import os
import src.utils.checker_util as ch
from src.controllers.detection_controller import YOLO_img_to_base64_response as yolo_b64
from flask import Flask, request, Response, jsonify, send_from_directory, abort
from flask_cors import CORS

folder = os.path.abspath('.') + "/weights"


def get_image():
	ch.Weight_checker.start(folder)
	return yolo_b64.predict(request.files["images"])


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/image", methods=["GET", "POST"])
def get():
	try:
		return Response(response=get_image(), status=200, mimetype="image/png")
	except FileNotFoundError:
		abort(404)





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
