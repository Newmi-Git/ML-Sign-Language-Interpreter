import os
from types import SimpleNamespace as NS

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from features import frame_vec, resample
from model import WordModel

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)   # when you deploy, use CORS(app, origins=["https://your-site"])

ck = torch.load(os.path.join(HERE, "models", "words.pt"), weights_only=True)
labels = ck["labels"]
net = WordModel(len(labels))
net.load_state_dict(ck["state"])
net.eval()


def lm(points):
    return [NS(x=p[0], y=p[1], z=p[2]) for p in points] if points else None


@app.get("/labels")
def get_labels():
    return jsonify(labels=labels)


@app.post("/predict")
def predict():
    frames = [frame_vec(lm(f["left"]), lm(f["right"]), lm(f["pose"]))
              for f in request.json["frames"]]
    if len(frames) < 10:
        return jsonify(label=None, conf=0)
    x = torch.tensor(resample(frames)).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(net(x), dim=1)[0]
    conf, i = p.max(0)
    return jsonify(label=labels[i.item()], conf=float(conf))


@app.post("/sentence")
def sentence():
    glosses = " ".join(request.json["glosses"])
    return jsonify(text=glosses.capitalize() + ".")           # fallback: just show the glosses


if __name__ == "__main__":
    app.run(port=5000)