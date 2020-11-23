from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_jwt import JWT, jwt_required

# from security import authenticate, identity
# from user import UserRegister


app = Flask(__name__)
# app.secret_key = 'kNjsvnoirDFHh9goimwD43itgnvqE#v8g8C!r@$#tge4fsfsfVst$#%$u*(tgJUsese6Lgsgsf'
api = Api(app)


# jwt = JWT(app, authenticate, identity)  # auth

items = []


class Item(Resource):
    @jwt_required()
    def get(self, name):
        item = next(filter(lambda x: x["name"] == name, items), None)
        return {'item': item}, 200 if item else 404

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
    def get(self):
        if len(items) > 0:
            return {"items": items}
        return {'items': None}, 404


api.add_resource(Item, '/item/<string:name>')
api.add_resource(Items, '/items')
# api.add_resource(UserRegister, '/register')


app.run(port=5000, debug=True)
