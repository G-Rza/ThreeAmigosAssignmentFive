#!/usr/bin/env python
# coding: utf-8

# In[3]:


# basehandler.py
import tornado.web

class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, clf, turi_accuracy, sklearn_accuracy):
        self.clf = clf
        self.turi_accuracy = turi_accuracy
        self.sklearn_accuracy = sklearn_accuracy

    def get_int_arg(self, name, default=None):
        try:
            return int(self.get_argument(name))
        except (ValueError, TypeError):
            return default

    def write_json(self, data):
        self.set_header("Content-Type", "application/json")
        self.write(data)


# In[ ]:




