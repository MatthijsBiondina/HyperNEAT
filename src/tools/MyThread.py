import threading

from tools.tools import log

class MyThread(threading.Thread):
    def __init__(self, threadID, name, function):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name     = name
        self.function = function

    def run(self):
        log(self,"Starting Thread " + self.name)
        self.function()
        log(self,"Closing Thread " + self.name)
