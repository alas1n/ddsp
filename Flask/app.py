from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse
from flask_jwt import JWT, jwt_required

# from security import authenticate, identity
# from user import UserRegister


app = Flask(__name__)
api = Api(app)



items = []


class Item(Resource):
    @jwt_required()
    def get(self, name):
        item = next(filter(lambda x: x["name"] == name, items), None)
        return {'item': item}, 200 if item else 404
        # return "test"

    def post(self, name):
        if next(filter(lambda x: x["name"] == name, items), None):
            return {'message': "An item with name '{}' already exist".format(name)}, 400
        data = request.get_json()
        item = {"name": name, "price": data['price']}
        items.append(item)
        return item, 201

    # def delete(self, name):
    #     global items
    #     items = list(filter(lambda x: x['name'] != name, items))
    #     return {"message": "item deleted"}

    # def put(self, name):
    #     # data = request.get_json()
    #     parser = reqparse.RequestParser()
    #     parser.add_argument('price',
    #                         type=float,
    #                         required=True,
    #                         help="This field cannot be left balnk!")
    #     data = parser.parse_args()
    #     item = next(filter(lambda x: x["name"] == name, items), None)
    #     if item is None:
    #         item = {"name": name,  "price": data["price"]}
    #         items.append(item)
    #     else:
    #         item.update(data)
    #     return item


class Items(Resource):
    def get(self,name):
        # if len(items) > 0:
        #     return {"items": items}
        return items
        # for item in items:
            # return item
    def post(self,name):
        if next(filter(lambda x: x["name"] == name, items), None):
            return {'message': "An item with name '{}' already exist".format(name)}, 400
        data = request.get_json()
        item = {"name": name}
        items.append(item)
        # return item, 201
        return item
        # return "post test"

@app.route('/audiorecog', methods = ['GET', 'POST'])
def audiorecog():
   if request.method == 'POST':
      print("Recieved Audio File")
      file = request.files['file']
      print('File from the POST request is: {}'.format(file))
      with open("audio.wav", "wb") as aud:
            aud_stream = file.read()
            aud.write(aud_stream)
      return "Success"
   else:
       return send_file('audio.wav', attachment_filename='audio.wav')



api.add_resource(Item, '/item/<string:name>')
api.add_resource(Items, '/items/<string:name>')
# api.add_resource(UserRegister, '/register')


app.run(port=5000, debug=True)
