import math
import os
import random
import signal
import weakref
import numpy as np
import time
import threading

import psutil
from vpython import sphere, vector, color, scene, rate, label, text, wtext
from vpython.no_notebook import stop_server

from sim_src.core.env_object import EnvObject, EnvObjectRunnable

class VisObject(EnvObject):
    _instances = weakref.WeakSet()
    def __init__(self, *args, **kwargs):
        self.__class__._instances.add(self)
        self.plot_obj = None
        self.vis_update_counter = 0 
        # self.env.process(self.vis_update_process())
        self.vis_update_interval_us = 1000
    
    def upd_vis_object(self):
        '''
        This function should be implemented in the child class
        For example: a satellite object should update its position and orientation separately, 
        then this function will load the current position and orientation to the plot_obj
        '''
        print("This function should be implemented in the child class")
        pass

    # def vis_update_process(self):
    #     while True:
    #         start = time.time()
    #         self.upd_vis_object()
    #         self.vis_update_counter += 1
    #         yield self.env.timeout(self.vis_update_interval_us)
    #         elapsed = time.time()
    #         time_us_per_tic = (elapsed - start)*1e6/self.vis_update_interval_us
    #         self.vis_update_interval_us = int ((1 / self.FRAME_RATE)*1e6 / time_us_per_tic)
    #         self.vis_update_interval_us = max(1,self.vis_update_interval_us)
    #         self.vis_update_interval_us = max(30000,self.vis_update_interval_us)
    #         # print("vis_update_interval_us",self.vis_update_interval_us,self.env.now)
    #         # print("vis_update_counter",self.vis_update_counter,self.vis_update_counter/EnvObject.get_run_time_us()*1e6)

class VisualSystem(EnvObject, threading.Thread):
    frame_rate = 30
    min_range = 1.1
    max_range = 1.5

    def __init__(self) -> None:
        super().__init__()
        # Configure the VPython scene:
        scene.width = 800
        scene.height = 600
        scene.background = color.gray(0.3)
        scene.fov = 0.8
        scene.up = vector(0, 1, 0)
        scene.ambient = color.white
        scene.lights = []
        self.my_text = wtext(text="Initial wtext value.")
        self.start()  # Start the visualization thread

    def run(self):
        print("Visual system started")
        while True:
            if EnvObject.is_end or not threading.main_thread().is_alive():
                print("Visual system stopping")
                stop_server()
                break
            # Cap the update rate
            rate(self.frame_rate)
            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            self.my_text.text = (
                f"Real Time: {EnvObject.get_run_time_us()/1e3:15.0f} ms, "
                f"Sim Time: {EnvObject.env.now/1e3:15.0f} ms, "
                f"Memory: {memory_bytes/1e6:15.0f} MB"
            )
            # Iterate over all active VisObject instances via the WeakSet:
            vis_objects = random.sample(VisObject._instances, k=len(VisObject._instances))
            max_update_time = 1.0 / self.frame_rate * 0.1  # maximum allowed update time (half a frame's duration)
            start_time = time.time()
            for vis_obj in vis_objects:
                try:
                    vis_obj.upd_vis_object()
                except Exception as e:
                    print("Error in visual update:", e)
                if time.time() - start_time > max_update_time:
                    print("Warning: Visual update cycle exceeded allowed time; skipping remaining updates.")
                    break
            # Adjust scene range if needed:
            if scene.range < self.min_range:
                scene.range += max(0.02, 0.1*(self.min_range - scene.range))
            if scene.range > self.max_range:
                scene.range -= max(0.02, 0.1*(scene.range - self.max_range))

    @classmethod
    def started(cls):
        return list(cls._instances)
    
    
if __name__ == "__main__":
    from sim_src.core.env_object import EnvObjectRunnable
    class test_o(VisObject):
        def upd_vis_object(self):
            if not self.plot_obj:
                print("No plot_obj")
                self.plot_obj = sphere(pos=vector(0, 0, 0), radius=1,
                    texture="simple_earth_texture.png",emissive=True)
                return self.plot_obj
            else:
                # pass
                self.plot_obj.rotate(angle=0.1, axis=vector(0, 1, 0))
                
    EnvObject.init_env()
    to = test_o()   
    vs = VisualSystem()
    EnvObject.run(until=200000000)
    
