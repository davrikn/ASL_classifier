import re
from flask import Flask, request, abort, jsonify

app = Flask(__name__)
app.config.update()

@app.route('/health', methods=['GET'])
def health():
    return "OK"

@app.route('/predict', methods=["GET"])
def predict_data():
    image = request.args.get("image", '')
    if image == '':
        abort(400, "Missing image parameter")
    if not re.match("^\d+,\d+,\d+$", image):
        abort(400, "Invalid input, must be rgb values separated by commas")
    [r,g, b] = re.split(",", image)
    if len(r) != len(g) or len(r) != len(b):
        abort(400, "Invalid input, RGB dimensions differ")
    if len(r) % 3 is not 0:
        abort(400, "Invalid input, RGB chains must be divisible by 3")

    if len(r) is not 3*200*200:
        abort(400, "Input must be 200x200 RGB data")

    r = list(map(int, [r[i:i+3] for i in range(0, len(r), 3)]))
    r = list([r[i:i+200] for i in range(0, len(r), 200)])
    g = list(map(int, [g[i:i+3] for i in range(0, len(g), 3)]))
    g = list([g[i:i+200] for i in range(0, len(g), 200)])
    b = list(map(int, [b[i:i+3] for i in range(0, len(b), 3)]))
    b = list([b[i:i+200] for i in range(0, len(b), 200)])

    # TODO add model and predict
    return jsonify(dict({"r": r, "g": g, "b": b}))

