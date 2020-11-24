from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse
from flask_jwt import JWT, jwt_required

# from security import authenticate, identity
# from user import UserRegister


app = Flask(__name__)
api = Api(app)

class AudioRecorder(Resource):
    def post(self):
        print("audio File received")
        with open("audio.wav", "wb") as aud:
                aud_stream = request.files['file'].read()
                aud.write(aud_stream)
        return "Success"
    def get(self):
        # print("request.data",request.args)
        return send_file('audio.wav', attachment_filename='audio.wav')

# class AudioReciver(Resource):
#     def get(self):
#         print("request.data",request.args)
#         return send_file('audio.wav', attachment_filename='audio.wav')

# @app.route('/audioreciver/', methods=['GET',])
# def audioreciver():
#     print("request",request.args.get)
#     return send_file('audio.wav', attachment_filename='audio.wav')


api.add_resource(AudioRecorder, '/audiorecorder/')
# api.add_resource(AudioReciver, '/audioreciver/')


app.run(port=5000, debug=True)
