# -*- coding: utf-8 -*-
"""
Created on Tue May 14 00:22:07 2024

@author: balazs
"""

import taichi as ti
from acoustic_simulator.simulator_3D import (AcousticModel,
                                             Material)
from acoustic_simulator.lock_in import Lock_in_amplifier

@ti.kernel 
def update_states(state1: ti.template(),
                  state2: ti.template(),
                  model: ti.template(),
                  coupling_function: ti.template()):
    w, h, d = state1.P.shape
    for i, j, k in ti.ndrange(w, h, d):
        if model[i, j, k]:
            state1.cr[i, j, k] = coupling_function(state2.P[i, j, k])

class CoupledModel:
    def __init__(self, size, width, frame_size, base_material1,  model_material1, A1, f1,
                 base_material2,  model_material2, A2, f2, coupling):
        self.md1 = AcousticModel(size, width, frame_size, base_material1, model_material1, A1, f1)
        Sc = base_material2.c*self.md1.dt/self.md1.dx
        self.md2 = AcousticModel(size, width, frame_size, base_material2, model_material2, A2, f2, Sc)
        self.coupling_function = coupling
    
    def block(self, ceneter, w, h, d):
        self.md1.block(ceneter, w, h, d)
        self.md2.block(ceneter, w, h, d)
    
    def cylinder(self, ceneter, r, h):
        self.md1.cylinder(ceneter, r, h)
        self.md2.cylinder(ceneter, r, h)
    
    @property
    def P_detected(self):
        return self.md1.P_detected
    
    def couple(self):
        update_states(self.md1.state, self.md2.state, self.md2.model_ti, self.coupling_function)
    
    def update(self):
        self.md2.update()
        self.couple()
        self.md1.update()

    def display(self, iterations = 0, detect = False, normalise = True):
        self.md1.display(iterations=iterations, detect = detect, update_func=self.update, normalise=normalise)
    
    def simulate(self, simulated_time, save_interval, graphical = True, save = True, update_func = None, normalise = True):
        metadata = {'f0': [self.md1.f0, self.md2.f0],
                    'A0': [self.md1.A0, self.md2.A0],
                    'c0': [self.md1.base_material.c, self.md2.base_material.c],
                    'c1': [self.md1.model_material.c, self.md2.model_material.c],
                    'rho0': [self.md1.base_material.rho, self.md2.base_material.rho],
                    'rho1': [self.md1.model_material.rho, self.md2.model_material.rho]}
        self.md1.simulate(simulated_time, save_interval, graphical, save, update_func=self.update, metadata=metadata, normalise = normalise)