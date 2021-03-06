import flask
import googleapiclient.discovery
import numpy as np
import os, json
from numpy import array
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
    instances = json.loads(request.args.get('instances'))
    version = request.args.get('version')

    x = predict_json(project, region, model, instances, version)
    return jsonify(x[0][1])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
