#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python
'''Starts and runs the scikit learn server'''

from pymongo import MongoClient
import tornado.web
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from motor import motor_tornado  # Import Motor for asynchronous MongoDB connection
from pprint import PrettyPrinter

# Custom imports
from basehandler import BaseHandler
import turihandlers as th
import examplehandlers as eh

define("port", default=8000, help="run on the given port", type=int)

pp = PrettyPrinter(indent=4)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/[/]?", BaseHandler),
            (r"/Handlers[/]?", th.PrintHandlers),
            (r"/AddDataPoint[/]?", th.UploadLabeledDatapointHandler),
            (r"/GetNewDatasetId[/]?", th.RequestNewDatasetId),
            (r"/UpdateModel[/]?", th.UpdateModelForDatasetIdTuri),  # Updated handler for Turi
            (r"/PredictOne[/]?", th.PredictOneFromDatasetIdTuri),   # Updated handler for Turi
            (r"/GetExample[/]?", eh.TestHandler),
            (r"/DoPost[/]?", eh.PostHandlerAsGetArguments),
            (r"/PostWithJson[/]?", eh.JSONPostHandler),
            (r"/MSLC[/]?", eh.MSLC),
        ]

        self.handlers_string = str(handlers)

        try:
            print('=================================')
            print('====ATTEMPTING MONGO CONNECT=====')
            self.client = motor_tornado.MotorClient()  # Initialize MotorClient for asynchronous MongoDB connection
            print(self.client.server_info())  # Force pymongo to look for possible running servers, error if none running
            # If we get here, at least one instance of pymongo is running
            self.db = self.client.turidatabase  # Database with labeledinstances, models

        except Exception as e:
            print('Could not initialize database connection, stopping execution')
            print(f'Error: {e}')

        self.clf = {}  # The classifier model
        print('=================================')
        print('==========HANDLER INFO===========')
        pp.pprint(handlers)

        settings = {'debug': True}
        tornado.web.Application.__init__(self, handlers, **settings)

    def __exit__(self):
        self.client.close()  # Just in case


def main():
    tornado.options.parse_command_line()
    http_server = HTTPServer(Application(), xheaders=True)
    http_server.listen(options.port)
    IOLoop.instance().start()

if __name__ == "__main__":
    main()

