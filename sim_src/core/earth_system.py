import math
from sim_src.core.env_object import EnvObjectRunnable, EnvObject
from sim_src.core.visual_system import VisObject
from vpython import sphere, vector, color
class earth(EnvObjectRunnable,VisObject):
    ROTAION_PERIOD = 86164.0905
    earth_update_interval_us = 1000000
    def __init__(self):
        super().__init__()
        self.angle = 0             
    
    def run(self):
        while True:
            yield self.env.timeout(self.earth_update_interval_us)
            self.angle += 2*math.pi/self.ROTAION_PERIOD/1e6 * self.earth_update_interval_us

    def upd_vis_object(self):
        if not self.plot_obj:
            self.plot_obj = sphere(pos=vector(0, 0, 0), radius=1,
                texture="simple_earth_texture.png",emissive=True)
            return self.plot_obj
        else:
            new_axis = vector(1, 0, 0).rotate(angle=self.angle, axis=vector(0, 1, 0))
            new_up   = vector(0, 1, 0).rotate(angle=self.angle, axis=vector(0, 1, 0))
            # print(new_axis, self.angle)
            # Set these vectors to the sphere. This sets its orientation absolutely.
            self.plot_obj.axis = new_axis
            self.plot_obj.up   = new_up
            
if __name__ == "__main__":
    h = earth.ROTAION_PERIOD/60/60
    print(f"Earth rotates once every {h} hours.")
    
    from sim_src.core.visual_system import visual_system
    EnvObject.init_env(RT=True)
    et = earth()
    # to = orbit()   
    vs = visual_system()
    EnvObject.run(until=2000000000000)
    