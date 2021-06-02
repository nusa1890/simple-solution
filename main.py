import flask
import googleapiclient.discovery
import numpy as np
import os, json
from google.api_core.client_options import ClientOptions
from flask import Flask, request, jsonify

def predict_json(project, region, model, instances, version=None):
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/nusa/Documents/euphoric-fusion-312609-e20f53f87718.json"

    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=["GET"])
def index():
    project = request.args.get('project')
    region = request.args.get('region')
    model = request.args.get('model')
    instances = [[[0.24157303, 0.29166667, 0.2238806, 0.54140127, 0.19047619, 0.54140127], [0.34831461, 0.41666667, 0.41044776, 0.57961783, 0.23809524, 0.57961783]]]
    version = request.args.get('version')

    x = predict_json(project, region, model, instances, version)
    return jsonify(x[0])

app.run()

"""pm10, so2, co, o3, no2 = np.dsplit(x, 5)
print(pm10)
print(so2)
print(co)
print(o3)
print(no2)"""
