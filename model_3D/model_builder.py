# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:23:52 2024

@author: balazs
"""

import numpy as np
import taichi as ti
import taichi.math as tm
from acoustic_simulator.model_3D.transform import (render_model_homogenious_light,
                                                   render_content_homogenious_light,
                                                   cartesian_to_spherical_py,
                                                   spherical_to_cartesian_py)

from acoustic_simulator.model_3D.geometry import (part_ids,
                                                  in_cylinder,
                                                  add_cylinder,
                                                  in_block,
                                                  add_block,
                                                  in_model)
import matplotlib.pyplot as plt
from matplotlib import colormaps
cmap = colormaps['viridis']

class Model:
    def __init__(self, shape, extents, frame_shape):   
        self.model = np.zeros(shape, bool)
        self.model_ti = ti.field(ti.u1, shape)
        
        
        self.extents = extents
        self.extents_ti = ti.Struct(xmin = extents[0][0],
                                    xmax = extents[0][1],
                                    ymin = extents[1][0],
                                    ymax = extents[1][1],
                                    zmin = extents[2][0],
                                    zmax = extents[2][1])
        
        self.dx = (extents[0][1] - extents[0][0])/shape[0]
        self.dy = (extents[1][1] - extents[1][0])/shape[1]
        self.dz = (extents[2][1] - extents[2][0])/shape[2]
        
        self.frame = np.zeros(frame_shape)
        self.frame_ti = ti.field(float, frame_shape)
        
        self.camera_pos = tm.vec3((extents[0][0] + extents[0][1])/2, 
                                  10*extents[1][0]-9*extents[1][1],
                                  (extents[2][0] + extents[2][1])/2)
        
        self.look_at = tm.vec3((extents[0][0] + extents[0][1])/2, 
                               (extents[1][0] + extents[1][1])/2,
                               (extents[2][0] + extents[2][1])/2)
        
        self.parts_p = ti.field(ti.f64, shape = (100, 8))
        self.parts_n = 0
        

        self.field_of_vision = 2
        
        self.window = None
        self.canvas = None
        
        self.first_click = None
        self.second_click = None
        self.first_cam_pos = None
        
    
    def cylinder(self, center, r, h):
        add_cylinder(self.model_ti, self.extents_ti, tm.vec3(*center), r, h)
        self.model = self.model_ti.to_numpy()
        self.parts_p[self.parts_n, 0] = 0
        self.parts_p[self.parts_n, 1] = center[0]
        self.parts_p[self.parts_n, 2] = center[1]
        self.parts_p[self.parts_n, 3] = center[2]
        self.parts_p[self.parts_n, 4] = r
        self.parts_p[self.parts_n, 5] = h
        
        self.parts_n += 1
    
    def block(self, center, w, h, d):
        add_block(self.model_ti, self.extents_ti, tm.vec3(*center), w, h, d)
        self.model = self.model_ti.to_numpy()
        self.parts_p[self.parts_n, 0] = 1
        self.parts_p[self.parts_n, 1] = center[0]
        self.parts_p[self.parts_n, 2] = center[1]
        self.parts_p[self.parts_n, 3] = center[2]
        self.parts_p[self.parts_n, 4] = w
        self.parts_p[self.parts_n, 5] = h
        self.parts_p[self.parts_n, 6] = d
        
        self.parts_n += 1
    
    def render(self):
        self.frame_ti.fill(0)
        render_model_homogenious_light(self.parts_p, self.parts_n,
                                       self.model_ti,
                                       self.extents_ti,
                                       self.frame_ti,
                                       self.camera_pos,
                                       self.look_at,
                                       self.field_of_vision)
        self.frame = self.frame_ti.to_numpy()
    
    def show(self):
        self.render()
        plt.imshow(self.frame)
    
    def display(self, cmap = cmap):
        if not self.window:
            self.window = ti.ui.Window('Model', self.frame.shape)
            self.canvas = self.window.get_canvas()
        try:
            while self.window.running:
                if self.window.is_pressed(ti.ui.LMB):
                    if not self.first_click:
                        self.first_click = self.window.get_cursor_pos()
                        self.first_cam_pos = self.camera_pos[:]
                    self.second_click = self.window.get_cursor_pos()
                    if self.first_click:
                        delta_phi =  (self.second_click[0] - self.first_click[0])*self.field_of_vision
                        delta_theta = (self.second_click[1] - self.first_click[1])*self.field_of_vision
                        
                        r, phi, theta = cartesian_to_spherical_py(self.first_cam_pos, self.look_at)
                        phi += delta_phi
                        theta -= delta_theta
                        
                        X = spherical_to_cartesian_py(tm.vec3(r, phi, theta))
                        self.camera_pos = self.look_at + X                    
                else:
                    self.first_click = False
                    self.first_cam_pos = self.camera_pos[:]
                self.render()
                self.canvas.set_image(cmap(self.frame).astype(np.float32))
                self.window.show()
            else:
                self.window.destroy()
                self.window = None
        except KeyboardInterrupt:
            pass
            
            