#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# motorhandler.py
#!/usr/bin/python

from pymongo import MongoClient
import tornado.web
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from basehandler import BaseHandler  # Add this line
from motor import motor_tornado  # Add this line

import turicreate as tc
import json
import numpy as np

define("port", default=8000, help="run on the given port", type=int)

class MotorHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        super(MotorHandler, self).__init__(*args, **kwargs)
        # Initialize Motor client for asynchronous MongoDB operations
        self.client = motor_tornado.MotorClient()
        # Database with labeledinstances, models
        self.db = self.client.turidatabase

    def post(self):
        '''Save data point and class label to the database'''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess = data['dsid']

        dbid = yield self.db.labeledinstances.insert_one(
            {"feature": fvals, "label": label, "dsid": sess}
        )
        self.write_json({"id": str(dbid.inserted_id),
                         "feature": [str(len(fvals)) + " Points Received",
                                     "min of: " + str(min(fvals)),
                                     "max of: " + str(max(fvals))],
                         "label": label})

    def get(self):
        '''Get a new dataset ID for building a new dataset'''
        a = yield self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a is None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid']) + 1
        self.write_json({"dsid": newSessionId})

    def put(self):
        '''Train a new model (or update) for a given dataset ID'''
        dsid = self.get_int_arg("dsid", default=0)
        model_type = self.get_argument("model_type", default="default")

        data = yield self.get_features_and_labels_as_SFrame(dsid)

        # fit the model to the data
        turi_acc = -1  # defines Turi accuracy for later
        best_model = 'unknown'
        if len(data) > 0:
            if model_type == "xgboost":
                model = tc.classifier.create(data, target='target', model='xgboost_classifier', verbose=0)
            else:
                model = tc.classifier.create(data, target='target', model='logistic_classifier', verbose=0)

            yhat = model.predict(data)
            turi_acc = sum(yhat == data['target']) / float(len(data))

            self.clf[dsid] = model
            # save model for use later, if desired
            yield model.save('../models/turi_model_dsid%d' % (dsid))

        self.turi_accuracy[dsid] = turi_acc  # store Turi accuracy

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy": turi_acc})

    def get_features_and_labels_as_SFrame(self, dsid):
        # create feature vectors from the database
        features = []
        labels = []
        cursor = self.db.labeledinstances.find({"dsid": dsid})
        while (yield cursor.fetch_next):
            a = cursor.next_object()
            features.append([float(val) for val in a['feature']])
            labels.append(a['label'])

        # convert to a dictionary for Turi Create
        data = {'target': labels, 'sequence': np.array(features)}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

    def get_features_as_SFrame(self, vals):
        # create feature vectors from array input
        # convert to a dictionary of arrays for Turi Create
        tmp = [float(val) for val in vals]
        tmp = np.array(tmp)
        tmp = tmp.reshape((1, -1))
        data = {'sequence': tmp}

        # send back the SFrame of the data
        return tc.SFrame(data=data)

class ModelComparisonResults(BaseHandler):
    def get(self):
        dsid = self.get_int_arg("dsid", default=0)

        turi_acc = self.turi_accuracy.get(dsid, -1)
        sklearn_acc = self.sklearn_accuracy.get(dsid, -1)

        if turi_acc == -1 or sklearn_acc == -1:
            self.write_json({"error": "Unavailable: {}".format(dsid)})
            return

        # accuracies' response
        response = {
            "Turi_Create_Accuracy": turi_acc,
            "Sklearn_Accuracy": sklearn_acc,
            "DSID": dsid
        }

        self.write_json(response)

