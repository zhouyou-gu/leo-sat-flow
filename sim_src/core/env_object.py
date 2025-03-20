from collections import deque
import math
from typing import Deque, List
import time
import numpy as np

from random import shuffle

import simpy
import simpy.rt


class EnvObject():
    env = None
    start_time = - math.inf
    is_start = False
    is_end = False
    
    time_scaling_on_one_second = 1e-6
    def init_env(RT=True,scaling=100):
        if RT:
            EnvObject.env = simpy.rt.RealtimeEnvironment(factor=EnvObject.time_scaling_on_one_second/scaling, strict=False)
        else:
            EnvObject.env = simpy.Environment()
            
    @classmethod
    def run(cls,until=None):
        cls.start_time = time.time()/1e-6
        cls.is_start = True
        print("EnvObject run")
        cls.env.run(until=until)
        print("EnvObject end",cls.get_run_time_us())
        cls.is_end = True

    @staticmethod
    def run_process(until=None):
        yield EnvObject.env.timeout(until)
        EnvObject.is_end = True
    
    @classmethod 
    def get_run_time_us(cls):
        return time.time()/1e-6 - cls.start_time
            
class EnvObjectRunnable(EnvObject):
    def __init__(self) -> None:
        super().__init__()
        EnvObject.env.process(self.run())
        
    def run(self):
        pass


if __name__ == "__main__":
    class Parent(EnvObjectRunnable):
        def __init__(self,c):
            super().__init__()
            self.c = c
            
        def run(self):
            self.c.hello_out()
            yield self.env.timeout(1)
            print("Parent run",self.env.now)

    class Child(EnvObjectRunnable):
        def __init__(self) -> None:
            super().__init__()
            self.counter = 0
            
        def run(self):
            pass
            while True:
                yield self.env.timeout(1)
                time.sleep(0.1)
                print("Child run")
        
        def hello_out(self):
            print("child hello_out +",self.env.now)
            self.env.process(self.hello(1.1))
            self.env.process(self.hello(1.2))

                    
        def hello(self,a):
            self.counter += 1
            print("child hello +",self.env.now,a,self.counter)
            yield self.env.timeout(a)
            print("child hello run time",self.get_run_time_us())
            print("child hello -",self.env.now,a,self.counter)
    
    EnvObject.init_env()
    
    p = Parent(Child())
        
    EnvObject.run(until=10000)
