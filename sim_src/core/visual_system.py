import math
import signal
import weakref
import numpy as np
import time
import threading

from vpython import sphere, vector, color, scene, rate, label, text
from vpython.no_notebook import stop_server

from sim_src.core.env_object import EnvObject, EnvObjectRunnable

class VisObject(EnvObject):
    FRAME_RATE = 30
    def __init__(self, *args, **kwargs):
        # self.__class__._instances.add(self)
        self.plot_obj = None
        self.vis_update_counter = 0 
        self.env.process(self.vis_update_process())
        self.vis_update_interval_us = 1
    
    def upd_vis_object(self):
        '''
        This function should be implemented in the child class
        For example: a satellite object should update its position and orientation separately, 
        then this function will load the current position and orientation to the plot_obj
        '''
        pass

    def vis_update_process(self):
        while True:
            start = time.time()
            self.upd_vis_object()
            self.vis_update_counter += 1
            yield self.env.timeout(self.vis_update_interval_us)
            elapsed = time.time()
            time_us_per_tic = (elapsed - start)*1e6/self.vis_update_interval_us
            # print("time_us_per_tic",time_us_per_tic)
            self.vis_update_interval_us = int ((1 / self.FRAME_RATE)*1e6 / time_us_per_tic)
            self.vis_update_interval_us = max(1,self.vis_update_interval_us)
            self.vis_update_interval_us = min(int ((1 / self.FRAME_RATE)*1e6),self.vis_update_interval_us)
            # print("vis_update_interval_us",self.vis_update_interval_us)
            print("vis_update_counter",self.vis_update_counter,self.vis_update_counter/EnvObject.get_run_time_us()*1e6)

class visual_system(EnvObject,threading.Thread):
    frame_rate = 30
    min_range = 1.1
    max_range = 1.5
    _instances = weakref.WeakSet()
    def __init__(self) -> None:
        super().__init__()
        self.start()
        scene.width = 800
        scene.height = 600
        scene.background = color.gray(0.3)
        scene.fov = 0.8  # Field-of-view in radians
        scene.up = vector(0, 1, 0)
        scene.ambient = color.black
        scene.lights = []
        my_text = text(text="Python Threading is ****", pos=vector(0, 0.35, 1),
                    align="center", height=0.1, depth=0, color=color.black)  
        my_text.up = vector(0, 1, -0.5)
    def run(self):
        print("visual_system run")
        start_time = time.time()
        while True:
            if EnvObject.is_end or threading.main_thread().is_alive() == False:
                print("visual_system stop",time.time()-start_time)
                stop_server()
                break
            rate(self.frame_rate)
            if scene.range < self.min_range:
                scene.range += max(0.02, 0.1*(self.min_range-scene.range)) 
            if scene.range > self.max_range:
                scene.range -= max(0.02, 0.1*(scene.range-self.max_range)) 

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
    vs = visual_system()
    EnvObject.run(until=200000000)
    
