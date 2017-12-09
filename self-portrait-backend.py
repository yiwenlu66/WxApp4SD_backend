import threading
import atexit
from flask import Flask, request, send_from_directory
from os import listdir
import os.path
import numpy as np
from PIL import Image
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2


DATA_DIR = 'data'
POOL_TIME = 0.5

q = []
processing = set()
finished = None
style_dict = {}
lock = threading.Lock()
thread = threading.Thread()

watermark = Image.open('watermark.png').resize((512, 512), Image.ANTIALIAS)
channels = []
stubs = []
for i in range(3):
    channels.append(implementations.insecure_channel('127.0.0.1', 9000 + i))
    stubs.append(
        prediction_service_pb2.beta_create_PredictionService_stub(channels[i]))


def create_rpc_callback(user_id):
    def callback(result_future):
        print('callback: {}'.format(user_id))
        if not result_future.exception():
            im = np.array(
                result_future.result().outputs['output_image'].float_val
            ).reshape([512, 512, 3])
            im = Image.fromarray(im.astype(np.uint8))
            im = im.convert('RGBA')
            im = Image.alpha_composite(im, watermark).convert('RGB')
            im.save(os.path.join(DATA_DIR, user_id + '_processed.png'))
            with lock:
                if user_id in processing:
                    processing.remove(user_id)
                finished.add(user_id)
    return callback


def create_app():
    app = Flask(__name__)

    def interrupt():
        global thread
        thread.cancel()

    def process():
        global finished, q
        try:
            if q:
                with lock:
                    user_id = q[0]
                    q = q[1:]
                    processing.add(user_id)
                im = Image.open(os.path.join(DATA_DIR, user_id + '_raw.png'))
                im = im.resize((512, 512), Image.ANTIALIAS)
                im = im.convert('RGB')
                request = predict_pb2.PredictRequest()
                request.model_spec.name = 'fast_style_transfer'
                request.model_spec.signature_name = 'predict_image'
                request.inputs['input_image'].CopyFrom(
                    tf.contrib.util.make_tensor_proto(
                        np.asarray(im, np.float32), shape=[1, 512, 512, 3]
                    )
                )
                result_future = stubs[style_dict[user_id]].Predict.future(
                    request, 30.0)
                result_future.add_done_callback(create_rpc_callback(user_id))
        except:
            thread = threading.Timer(POOL_TIME, process, ())
            thread.start()

    def process_start():
        # initialize finished set
        global finished
        finished = set()
        for filename in listdir(DATA_DIR):
            if filename.endswith('_processed.png'):
                finished.add(filename.split('_')[0])

        global thread
        thread = threading.Timer(POOL_TIME, process, ())
        thread.start()

    process_start()
    atexit.register(interrupt)
    return app

app = create_app()

@app.route('/upload/', methods=['POST'])
def upload():
    global q
    user_id = request.form['id']
    if user_id not in q:
        im = Image.open(request.files['data'])
        im.save(os.path.join(DATA_DIR, user_id + '_raw.png'))
        with lock:
            style_dict[user_id] = int(request.form['style'])
            q.append(user_id)
        return 'ok'
    return 'failure'

@app.route('/download/', methods=['GET'])
def download():
    global finished
    user_id = request.args['id']
    if user_id in finished:
        return send_from_directory(DATA_DIR, user_id + '_processed.png')


@app.route('/query/', methods=['GET'])
def query():
    global finished, q
    user_id = request.args['id']
    try:
        return str(list(q).index(user_id) + 1)
    except ValueError:
        if user_id in processing:
            return '0'
        if user_id in finished:
            return '-1'
        return '-2'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
