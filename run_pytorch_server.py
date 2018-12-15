
import io
from PIL import Image
import flask
from flask_cors import CORS
import api_utils


# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
CORS(app)

single_obj_det_model = None
multi_class_model = None

def load_single_obj_det_model():
    global single_obj_det_model
    single_obj_det_model = api_utils.load_single_obj_det_model()
    return

def load_multi_class_model():
    global multi_class_model
    multi_class_model = api_utils.load_multi_class_model()
    return

@app.route("/predict", methods=["POST"])
def predict():
    if(single_obj_det_model==None):
        load_single_obj_det_model()

    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            api_utils.delete_other_result_imgs('largest-item-bbox')

            image_str = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image_str))
            res_url = api_utils.test_model_on_img(image, single_obj_det_model)
            data['res_url'] = res_url
            data["success"] = True

    return flask.jsonify(data)    

@app.route("/predictMultiClass", methods=["POST"])
def predict_multi_class():
    if(multi_class_model==None):
        load_multi_class_model()

    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            api_utils.delete_other_result_imgs('multi-class')

            image_str = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image_str))
            res_url = api_utils.get_multi_class_labeled_image(image, multi_class_model)
            data['res_url'] = res_url
            data["success"] = True

    return flask.jsonify(data)    

@app.route("/detect_edge", methods=["POST"])
def detect_edge():
    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            api_utils.delete_other_result_imgs('sobel-operator')
            image_str = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image_str))
            res_url = api_utils.apply_sobel_operator_on_custom_image(image)
            data['res_url'] = res_url
            data["success"] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    load_single_obj_det_model()
    app.run(host='0.0.0.0', port=5000)
