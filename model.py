import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle
import requests, json


class Model(object):
    def __init__(self):
        # read the datasets
        self.chefmozparking = pd.read_csv("chefmozparking.csv")
        self.geoplaces = pd.read_csv("geoplaces2.csv")
        self.rating_final = pd.read_csv("rating_final.csv")

    def transform(self):
        # merge chefmozpaarking with geoplaces
        places = self.chefmozparking.merge(self.geoplaces, on="placeID", how="inner")
        # take placeID, userID and rating columns
        self.rating_final = self.rating_final.iloc[:, 0:3]
        # merging places with rating_final
        places = places.merge(self.rating_final, on="placeID", how="inner")
        # select necessary features
        places = places[
            ["price", "parking_lot", "smoking_area", "other_services", "dress_code", "accessibility", "rating"]]
        # modify some values to present as a column name
        places.parking_lot[places.parking_lot == "valet parking"] = "valet_parking"
        places.smoking_area[places.smoking_area == "not permitted"] = "not_permitted"
        places.smoking_area[places.smoking_area == "only at bar"] = "only_at_bar"
        # modify the dataset to work with numeric values
        places = pd.get_dummies(places, drop_first=True)
        # get the train and test splits
        X_train, X_val, y_train, y_val = train_test_split(places.drop(["rating"], axis=1),
                                                          places.rating, test_size=0.001,
                                                          random_state=0)
        # build the model
        model = RandomForestClassifier(max_depth=3, random_state=0)
        # fit the model
        model.fit(X_train, y_train)
        # model serialization
        pickle.dump(model, open("random_forest.pkl", "wb"))
        return X_train, X_val, y_train, y_val

    def get_prediction(self):
        # get the splits
        splits = self.transform()
        # prepare the test vector
        X = splits[0].iloc[1, :].to_json(orient='split')
        # covert to json
        X = json.loads(X)
        # requested url
        BASE_URL = "http://localhost:8080"
        # send the post request and get the response
        response = requests.post("{}/predict".format(BASE_URL), json=X)
        return response.json()


