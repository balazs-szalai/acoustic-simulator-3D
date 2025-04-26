# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:46:01 2024

@author: balazs
"""
import taichi as ti
if __name__ == '__main__':
    ti.init(ti.gpu)

from . import model_3D

Model = model_3D.Model
render_content_homogenious_light = model_3D.render_content_homogenious_light
cartesian_to_spherical_py = model_3D.cartesian_to_spherical_py
spherical_to_cartesian_py = model_3D.spherical_to_cartesian_py

from . import funcs_3D 
detect = funcs_3D.detect
detect_in_space = funcs_3D.detect_in_space
step = funcs_3D.step
apply_filter = funcs_3D.apply_filter
SFs13p = funcs_3D.SFs13p


import taichi.math as tm
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
import cv2
from subprocess import Popen
cmap = colormaps['viridis']

class Material:
    def __init__(self, c, rho):
        self.c = c
        self.rho = rho

@ti.kernel 
def update_model_material(state: ti.template(),
                          model: ti.template(),
                          c0: float,
                          c: float,
                          rho: float):
    for i, j, k in ti.ndrange(*model.shape):
        if model[i, j, k]:
            state.cr[i, j, k] = c/c0
            state.rho[i, j, k] = rho

class AcousticModel(Model):
    def __init__(self, size, width, frame_size, base_material, model_material, A0, f0, Sc = 0.5):
        extents = [[0, width],
                   [0, width*size[1]/size[0]],
                   [0, width*size[2]/size[0]]]
        super().__init__(size, extents, frame_size)
        
        self.base_material = base_material
        self.model_material = model_material
        
        self.state = ti.Struct(vx = ti.field(float, shape = size),
                                vy = ti.field(float, shape = size),
                                vz = ti.field(float, shape = size),
                                P  = ti.field(float, shape = size),
                                rho = ti.field(float, shape = size),
                                cr = ti.field(float, shape = size))
        
        self.state.rho.fill(self.base_material.rho)
        self.state.cr.fill(1)
        
        self.temp = ti.Struct(vx = ti.field(float, shape = size),
                              vy = ti.field(float, shape = size),
                              vz = ti.field(float, shape = size),
                              P  = ti.field(float, shape = size))

        self.dx = (extents[0][1] - extents[0][0])/size[0]
        self.dy = self.dx
        self.dz = self.dx
        self.dt = Sc*self.dx/self.base_material.c
        self.Sc = Sc
        
        self.t = 0
        self.A0 = A0
        self.f0 = f0
        self.phi = 0

        self.k1 = ti.Struct(vx = ti.field(float, shape = size),
                            vy = ti.field(float, shape = size),
                            vz = ti.field(float, shape = size),
                            P  = ti.field(float, shape = size))
        self.k2 = ti.Struct(vx = ti.field(float, shape = size),
                            vy = ti.field(float, shape = size),
                            vz = ti.field(float, shape = size),
                            P  = ti.field(float, shape = size))
        self.k3 = ti.Struct(vx = ti.field(float, shape = size),
                            vy = ti.field(float, shape = size),
                            vz = ti.field(float, shape = size),
                            P  = ti.field(float, shape = size))
        self.k4 = ti.Struct(vx = ti.field(float, shape = size),
                            vy = ti.field(float, shape = size),
                            vz = ti.field(float, shape = size),
                            P  = ti.field(float, shape = size))
        
        self.source = ti.Struct(ixmin = 0,
                                ixmax = 0,
                                iymin = 0,
                                iymax = 0,
                                izmin = 0,
                                izmax = 0)
        
        self.detector = ti.Struct(ixmin = 0,
                                  ixmax = 0,
                                  iymin = 0,
                                  iymax = 0,
                                  izmin = 0,
                                  izmax = 0)
        self.P_detected = []
    
    def render(self, content = None):
        self.frame_ti.fill(0)
        if not content:
            content = self.state.P
        render_content_homogenious_light(self.parts_p, self.parts_n,
                                         self.model_ti,
                                         self.extents_ti,
                                         content,
                                         self.frame_ti,
                                         self.camera_pos,
                                         self.look_at,
                                         self.field_of_vision)
        self.frame = self.frame_ti.to_numpy()
    
    def cylinder(self, center, r, h):
        super().cylinder(center, r, h)
        update_model_material(self.state, self.model_ti,
                              self.base_material.c, self.model_material.c,
                              self.model_material.rho)
    
    def block(self, center, w, h, d):
        super().block(center, w, h, d)
        update_model_material(self.state, self.model_ti,
                              self.base_material.c, self.model_material.c,
                              self.model_material.rho)
    
    def set_detector(self, center, w, h, d):
        xmin = center[0] - w/2
        xmax = center[0] + w/2
        
        ymin = center[1] - h/2
        ymax = center[1] + h/2
        
        zmin = center[2] - d/2
        zmax = center[2] + d/2
        
        ixmin = int((xmin-self.extents[0][0])//self.dx)
        ixmax = int((xmax-self.extents[0][0])//self.dx)
        
        iymin = int((ymin-self.extents[1][0])//self.dy)
        iymax = int((ymax-self.extents[1][0])//self.dy)
        
        izmin = int((zmin-self.extents[2][0])//self.dz)
        izmax = int((zmax-self.extents[2][0])//self.dz)
        
        self.detector.ixmin = ixmin
        self.detector.ixmax = ixmax
        self.detector.iymin = iymin
        self.detector.iymax = iymax
        self.detector.izmin = izmin
        self.detector.izmax = izmax
    
    def set_source(self, center, w, h, d):
        xmin = center[0] - w/2
        xmax = center[0] + w/2
        
        ymin = center[1] - h/2
        ymax = center[1] + h/2
        
        zmin = center[2] - d/2
        zmax = center[2] + d/2
        
        ixmin = int((xmin-self.extents[0][0])//self.dx)
        ixmax = int((xmax-self.extents[0][0])//self.dx)
        
        iymin = int((ymin-self.extents[1][0])//self.dy)
        iymax = int((ymax-self.extents[1][0])//self.dy)
        
        izmin = int((zmin-self.extents[2][0])//self.dz)
        izmax = int((zmax-self.extents[2][0])//self.dz)
        
        self.source.ixmin = ixmin
        self.source.ixmax = ixmax
        self.source.iymin = iymin
        self.source.iymax = iymax
        self.source.izmin = izmin
        self.source.izmax = izmax
    
    def update(self):
        step(self.state,
             self.temp,
             self.k1,
             self.k2,
             self.k3,
             self.k4,
             self.source,
             self.A0,
             self.f0,
             self.base_material.c,
             self.Sc,
             self.t,
             self.phi)
        self.t += self.dt
        return self.t
    
    def simulate(self, simulated_time, save_interval, graphical = True, save = True, update_func = None, metadata = None, normalise = True):
        if update_func == None:
            update_func = self.update
        if isinstance(save, str):
            path = save[:] + f'/results_{time.strftime("%d_%m_%Y_%H_%M_%S")}'
        elif save:
            path = f'results_{time.strftime("%d_%m_%Y_%H_%M_%S")}'
        
        if save:
            os.makedirs(path, exist_ok=True)
        with tqdm(total=int(simulated_time//self.dt)) as pbar:
            try:
                j = 0
                if not metadata:
                    metadata = {'f0': self.f0,
                                'A0': self.A0,
                                'c0': self.base_material.c,
                                'c1': self.model_material.c,
                                'rho0': self.base_material.rho,
                                'rho1': self.model_material.rho}
                while self.t < simulated_time:
                    for i in range(int(save_interval//self.dt)):
                        update_func()
                    self.detect()
                    pbar.update(int(save_interval//self.dt))
                    if graphical:
                        self.render()
                        if normalise:
                            offset = max(abs(np.max(self.frame)), abs(np.min(self.frame)))
                            img = (cmap(((self.frame+offset).T/(2*offset)*255).astype(np.ubyte))*255).astype(np.ubyte)
                        else:
                            img = ((cmap(self.frame.T)*255).astype(np.ubyte))
                        # img = (cmap(self.frame.T)*255).astype(np.ubyte)
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        cv2.imwrite(os.path.join(path, f'acoustic_pressure_{j:06d}.png'), img)
                    # metadata.append(self.t)
                    j += 1
            finally:
                if save:
                    np.savetxt(os.path.join(path, 'P_detected.txt'), self.P_detected)
                    try:
                        np.save(os.path.join(path, 'metadata.npy'), metadata, allow_pickle=True)
                    except:
                        pass
                    if graphical:
                        p = Popen(f'ffmpeg -loglevel panic -framerate {30} -i {path}\\acoustic_pressure_%06d.png -s:v {self.frame.shape[0]}x{self.frame.shape[1]} -c:v h264_nvenc -preset:v slow -qp:v 16 -pix_fmt yuv420p -y {path}\\{path}.mp4')
                        p.communicate() 
    
    def detect(self):
        P = detect(self.state,
                   self.detector)
        self.P_detected.append([self.t, P])
        return P
    
    def detect_in_space(self):
        detected = np.empty((self.detector.ixmax - self.detector.ixmin,
                             self.detector.iymax - self.detector.iymin,
                             self.detector.izmax - self.detector.izmin), np.float64)
        detect_in_space(self.state,
                        self.detector,
                        detected)
        return detected
    
    def display(self, cmap = cmap, iterations = 0, detect = False, update_func = None, normalise = False):
        if update_func == None:
            update_func = self.update
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
                        phi += delta_theta
                        theta -= delta_phi
                        
                        X = spherical_to_cartesian_py(tm.vec3(r, phi, theta))
                        self.camera_pos = self.look_at + X                    
                else:
                    self.first_click = False
                    self.first_cam_pos = self.camera_pos[:]
                    
                if iterations:
                    for i in range(iterations):
                        update_func()
                if detect:
                    self.detect()
                    print(self.P_detected[-1])
                self.render()
                if normalise:
                    offset = max(abs(np.max(self.frame)), abs(np.min(self.frame)))
                    self.canvas.set_image(cmap(((self.frame+offset)/(2*offset)*255).astype(np.ubyte)).astype(np.float32))#
                else:
                    self.canvas.set_image(cmap(self.frame).astype(np.float32))
                self.window.show()
            else:
                self.window.destroy()
                self.window = None
        except KeyboardInterrupt:
            pass       
    
    def show(self, content = None, ui = 'window'):
        self.render(content)
        if ui == 'plt':
            plt.imshow(self.frame)
        elif ui == 'window':
            if not self.window:
                self.window = ti.ui.Window('Model', self.frame.shape)
                self.canvas = self.window.get_canvas()
            offset = max(abs(np.max(self.frame)), abs(np.min(self.frame)))
            self.canvas.set_image(cmap(((self.frame+offset)/(2*offset)*255).astype(np.ubyte)).astype(np.float32))
            self.window.show()
    
    def clear(self):
        self.t = 0
        self.P_detected = []
        
        self.state.P.fill(0)
        self.state.vx.fill(0)
        self.state.vy.fill(0)
        self.state.vz.fill(0)
        
        self.temp.P.fill(0)
        self.temp.vx.fill(0)
        self.temp.vy.fill(0)
        self.temp.vz.fill(0)

        self.k1.P.fill(0)
        self.k1.vx.fill(0)
        self.k1.vy.fill(0)
        self.k1.vz.fill(0)
        self.k2.P.fill(0)
        self.k2.vx.fill(0)
        self.k2.vy.fill(0)
        self.k2.vz.fill(0)
        self.k3.P.fill(0)
        self.k3.vx.fill(0)
        self.k3.vy.fill(0)
        self.k3.vz.fill(0)
        self.k4.P.fill(0)
        self.k4.vx.fill(0)
        self.k4.vy.fill(0)
        self.k4.vz.fill(0)
        
#%%
if __name__ == '__main__':    
    brass = Material(3400, 7800)
    air = Material(340, 1.5)    
    
    md = AcousticModel((50, 38, 75), 10/3, (800, 600), brass, air, 100, 168.0515)
    md.cylinder([5/3, 5/4, 5/2], 1, 2)
    md.cylinder([5/3, 5/4, 3.55], 0.5, 0.1)
    md.cylinder([5/3, 5/4, 1.45], 0.5, 0.1)
    md.block([5/3, 5/4, 4.1], 10, 1, 1)
    md.block([5/3, 5/4, 0.9], 10, 1, 1)
    md.set_detector([3, 5/4, 4.1], 0.1, 0.9, 0.9)
    md.set_source([1/3, 5/4, 0.9], 0.1, 0.9, 0.9)
    # md.set_source([1/3, 5/4, 0.9], 0.1, 0.9, 0.9)
#%%

    As = []
    for f in np.linspace(167, 170, 20):
        md.f0 = f
        md.clear()
        md.simulate(1.5, 0.0001, graphical=False, save = False)
        P = np.array(md.P_detected)
        t = P[:, 0]
        P = P[:, 1]
        P = P[t > 1]
        P_fft = np.fft.fft(P, len(P))
        fft_freq = np.fft.fftfreq(len(P), 0.0001)
        P_fft *= np.exp(-(fft_freq-f)**2/(2*2**2))*(1 + 1j)
        P = np.fft.ifft(P_fft)
        As.append([f, np.mean(np.abs(P))])
    As = np.array(As)
    plt.plot(As[:, 0], As[:, 1])
    # md.display(iterations=30, detect=True)
#%%
#     window = ti.ui.Window(name='Simple acoustic simulation', res=(800, 600), fps_limit=60, pos = (150, 150))
#     gui = window.get_canvas()
# #%%   
#     i = 0
#     while True:
#         try:
#             w = 3.2 + tm.sin(i/100)*2.8
#             # print(w)
#             md.camera_pos = tm.vec3(5+10*tm.cos(w), 5+10*tm.sin(w), 10*tm.sin(w))
#             md.render()
#             gui.set_image(cmap(md.frame).astype(np.float32))
#             window.show()
#             i += 1
#         except:
#             break