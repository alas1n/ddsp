from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse
from flask_jwt import JWT, jwt_required
from pydub import AudioSegment
from timbre_transfer_mine import soundTimberFunc
import os

app = Flask(__name__)
api = Api(app)


class AudioRecorder(Resource):
    def post(self):
        print("audio File received")
        with open("audio.mp3", "wb") as aud:
            aud_stream = request.files['file'].read()
            aud.write(aud_stream)
            sound = AudioSegment.from_mp3("audio.mp3")
            sound.export("audio.wav", format="wav")
            os.system("ffmpeg -i audio.wav -ar 16000 -ac 1 audio_16K_mono.wav" + " -y") 
        return "Success"

class TimbreTarnsfer(Resource):
    def get(self):
        print(request.args['num'])
        print(request.args['model'])
        model = request.args['model']
        soundTimberFunc('audio_16K_mono.wav',model)

        ResynthesisModel = "Resynthesis_" + model
        resynthesisFile = "%s.wav" % ResynthesisModel
        low = ResynthesisModel + "_low"
        lowresynthesisFile = "%s.wav" % low
        os.system("ffmpeg -i "+ resynthesisFile + " -ar 44100 -ac 1 " + lowresynthesisFile + " -y") 
        return send_file(lowresynthesisFile, attachment_filename=lowresynthesisFile)

api.add_resource(AudioRecorder, '/audiorecorder/')
api.add_resource(TimbreTarnsfer, '/timbretransfer/')


app.run(port=5000, debug=True)
