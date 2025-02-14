from collections import deque
from typing import Deque, List
import time
import numpy as np

from random import shuffle

import simpy


class EnvObject():
    env = None
    @classmethod
    def set_env(cls,env):
        cls.env = env

class EnvObjectRunnable(EnvObject):
    def __init__(self) -> None:
        EnvObject.env.process(self.run())
        self.init_time_us = time.time()/1e-6
        
    def run(self):
        pass
    
    def get_run_time_us(self):
        return time.time()/1e-6 - self.init_time_us

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
                print("Child run")
        
        def hello_out(self):
            print("child hello_out +",self.env.now)
            self.env.process(self.hello(1.1))
            self.env.process(self.hello(1.2))

                    
        def hello(self,a):
            self.counter += 1
            print("child hello +",self.env.now,a,self.counter)
            yield self.env.timeout(a)
            print("child hello -",self.env.now,a,self.counter)
            
    env = simpy.Environment()
    EnvObject.set_env(env=env)
    p = Parent(Child())
    env.run(until=10)