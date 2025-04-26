# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:00:32 2024

@author: balazs
"""

import mph
import numpy as np
import taichi as ti

ti.init()

client = mph.start()
model = client.load(r'C:\Users\balaz\Desktop\MFF_UK\SFG\akustické_a_tepelne_vlnenie_v_heliu\resonator_for_build.mph')

Xm = model.evaluate('Xm')[0]
Ym = model.evaluate('Ym')[0]
Zm = model.evaluate('Zm')[0]


# my_model = np.zeros(dtype = np.float64, shape = (300, 300, 300))
# model_extents = [[min(Xm[0]), max(Xm[0])], [min(Ym[0]), max(Ym[0])], [min(Zm[0]), max(Zm[0])]]

my_model = ti.field(ti.u1, shape=(300, 300, 300))
model_extents = ti.Struct(xmin = min(Xm),
                          xmax = max(Xm),
                          ymin = min(Ym),
                          ymax = max(Ym),
                          zmin = min(Zm),
                          zmax = max(Zm))

X = ti.field(ti.f64, shape=Xm.shape)
X.from_numpy(Xm)

Y = ti.field(ti.f64, shape=Ym.shape)
Y.from_numpy(Ym)

Z = ti.field(ti.f64, shape=Zm.shape)
Z.from_numpy(Zm)

@ti.func 
def coordinates_to_model_index(x: float, y: float, z: float, 
                               model: ti.template(), 
                               model_extents: ti.template()):
    x_len, y_len, z_len = model.shape
    
    dx = (model_extents.xmax - model_extents.xmin)/x_len
    dy = (model_extents.ymax - model_extents.ymin)/y_len
    dz = (model_extents.zmax - model_extents.zmin)/z_len
    
    ix = (x-model_extents.xmin)//dx
    iy = (y-model_extents.ymin)//dy
    iz = (z-model_extents.zmin)//dz
    
    return ti.Vector([ix, iy, iz])
    

@ti.kernel 
def get_model():
    for i in range(X.shape[0]):
        ix, iy, iz = coordinates_to_model_index(X[i], Y[i], Z[i], my_model, model_extents)
        my_model[int(ix), int(iy), int(iz)] = True

get_model()