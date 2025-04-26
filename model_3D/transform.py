# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:46:03 2024

@author: balazs
"""

import taichi as ti 
import taichi.math as tm
from acoustic_simulator.model_3D.geometry import (scalar,
                                                  vnorm,
                                                  is_boundary,
                                                  ray_box_intersection,
                                                  ray_cast)

@ti.func 
def sgn(x: float) -> float:
    ret = 0
    if x > 0:
        ret = 1
    if x < 0:
        ret = -1
    return ret

def vec_prod_f(u, v):
    w = [u[1]*v[2] - u[2]*v[1],
         u[2]*v[0] - u[0]*v[2],
         u[0]*v[1] - u[1]*v[0]]
    return tm.vec3(w)

@ti.func
def vec_prod(u, v):
    w = tm.vec3([u[1]*v[2] - u[2]*v[1],
                 u[2]*v[0] - u[0]*v[2],
                 u[0]*v[1] - u[1]*v[0]])
    return w

@ti.func 
def cartesian_to_spherical(X:tm.vec3, X0:tm.vec3 = tm.vec3(0,0,0)) -> tm.vec3:
    X -= X0
    r = tm.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    phi = sgn(X[1])*tm.acos(X[0]/tm.sqrt(X[0]**2 + X[1]**2))
    theta = tm.acos(X[2]/r)
    return tm.vec3(r, phi, theta)

@ti.func 
def get_matrix_of_rotation(n):
    lx = n[0]
    ly = n[1]
    
    alph1 = tm.atan2(ly, lx)
    
    M1 = tm.mat3([[ tm.cos(alph1), tm.sin(alph1), 0],
                  [-tm.sin(alph1), tm.cos(alph1), 0],
                  [             0,             0, 1]])
    
    n1 = M1@n
    
    lx = n1[0]
    lz = n1[2]
    
    alph2 = tm.atan2(lx, lz)
    
    M2 = tm.mat3([[tm.cos(alph2), 0, -tm.sin(alph2)],
                  [0            , 1,              0],
                  [tm.sin(alph2), 0, tm.cos(alph2)]])
    
    return M2@M1

@ti.func 
def get_matrix_of_inverse_rotation(n):
    lx = n[0]
    ly = n[1]
    
    alph1 = tm.atan2(ly, lx)
    
    M1 = tm.mat3([[ tm.cos(alph1), tm.sin(alph1), 0],
                  [-tm.sin(alph1), tm.cos(alph1), 0],
                  [             0,             0, 1]])
    Mi1 = tm.mat3([[tm.cos(alph1), -tm.sin(alph1), 0],
                   [tm.sin(alph1),  tm.cos(alph1), 0],
                   [            0,              0, 1]])
    
    n1 = M1@n
    
    lx = n1[0]
    lz = n1[2]
    
    alph2 = tm.atan2(lx, lz)
    
    Mi2 = tm.mat3([[tm.cos(alph2), 0, tm.sin(alph2)],
                  [0            , 1,              0],
                  [-tm.sin(alph2), 0, tm.cos(alph2)]])
    
    return Mi1@Mi2

@ti.func 
def proj(v, n):
    return v - n*scalar(n, v)

@ti.func 
def to_cam(a: tm.vec3,
           cam_pos: tm.vec3,
           look_dir: tm.vec3,
           d:float,
           M: tm.mat3):
    
    ac = a-cam_pos
    a_rot = M@ac
    
    ap = proj(a_rot, tm.vec3([0, 0, 1]))
    return tm.vec2([ap[0], ap[1]])

@ti.func 
def from_cam(a: tm.vec2,
             d:float,
             M: tm.mat3):
    
    ao = tm.vec3([a[0], a[1], d])
    a_rot = M@ao
    
    return a_rot/vnorm(a_rot)

# @ti.func 
# def point_to_ind(X: tm.vec2,
#                  xmin: float,
#                  ymin: float,
#                  dx: float,
#                  dy: float):
#     x, y = X
#     x += xmin
#     y += ymin
    
#     ix = int(x/dx)
#     iy = int(y/dy)
    
#     return tm.ivec2(ix, iy)

def sgn_py(x: float) -> float:
    ret = 0
    if x > 0:
        ret = 1
    if x < 0:
        ret = -1
    return ret

def cartesian_to_spherical_py(X:tm.vec3, X0:tm.vec3 = tm.vec3(0,0,0)) -> tm.vec3:
    X -= X0
    r = tm.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    phi = sgn_py(X[1])*tm.acos(X[0]/tm.sqrt(X[0]**2 + X[1]**2))
    theta = tm.acos(X[2]/r)
    return tm.vec3(r, phi, theta)

def spherical_to_cartesian_py(X:tm.vec3) -> tm.vec3:
    r, phi, theta = X
    x = r*tm.sin(theta)*tm.cos(phi)
    y = r*tm.sin(theta)*tm.sin(phi)
    z = r*tm.cos(theta)
    return tm.vec3(x, y, z)

@ti.func
def normal(i: int, j: int, k: int, model: ti.template()):
    T = tm.vec3(0,0,0)
    for l, m, p in ti.ndrange(3, 3, 3):
        if model[i-1+l, j-1+m, k-1+p]:
            T += tm.vec3(-1+l, -1+m, -1+p)
    T /= -vnorm(T)
    return T

@ti.func 
def homogenious_lighting(n: tm.vec3,
                         pos: tm.vec3,
                         my_pos: tm.vec3):
    
    w = (my_pos - pos)
    w /= vnorm(w)
    
    amplitude = scalar(w, n)
    
    if amplitude < 0:
        amplitude = 0.0 
    
    return amplitude


@ti.kernel
def render_model_homogenious_light(parts: ti.template(), parts_n: int,
                                   model: ti.template(),
                                   model_extents: ti.template(),
                                   frame: ti.template(),
                                   cam_pos: tm.vec3,
                                   look_at: tm.vec3,
                                   field_of_vision: float):
    look_dir = look_at-cam_pos
    d = vnorm(look_dir)
    look_dir /= d
    
    fw, fh = frame.shape
    r0 = max(fh, fw)
    rx, ry = fw/r0, fh/r0
    
    fovx, fovy = field_of_vision*rx, field_of_vision*ry
    
    xmin = d*tm.tan(fovx/2)
    ymin = d*tm.tan(fovy/2)
    
    M = get_matrix_of_rotation(look_dir)
    iM = get_matrix_of_rotation(M@tm.vec3([0,0,1]))
    
    dx = (model_extents.xmax - model_extents.xmin)/model.shape[0]
    dy = (model_extents.ymax - model_extents.ymin)/model.shape[1]
    dz = (model_extents.zmax - model_extents.zmin)/model.shape[2]
    
    w, h = frame.shape
    dX = 2*xmin/w
    dY = 2*ymin/h
    
    for i, j in ti.ndrange(*frame.shape):
        X = tm.vec2([-xmin + i*dX,
                     -ymin + j*dY])
        v = from_cam(X, d, iM)
        
        if ray_box_intersection(cam_pos, v, min(dx, dy, dz), model_extents):
            pos = ray_cast(cam_pos, v, min(dx, dy, dz),
                                parts, parts_n, model_extents)
            X, Y = to_cam(pos, cam_pos, look_dir, d, M)
            x, y, z = pos
            
            ix = int((x-model_extents.xmin)/dx)
            iy = int((y-model_extents.ymin)/dy)
            iz = int((z-model_extents.zmin)/dz)
            
            if (ix >= model.shape[0]-1 or ix < 1 or
                iy >= model.shape[1]-1 or iy < 1 or
                iz >= model.shape[2]-1 or iz < 1):
                frame[i, j] = 0
            else:
                n = normal(ix, iy, iz, model)
                
                frame[i, j] = homogenious_lighting(n, pos, cam_pos)
        else:
            frame[i, j] = 0

@ti.kernel
def render_content_homogenious_light(parts: ti.template(), parts_n: int,
                                     model: ti.template(),
                                     model_extents: ti.template(),
                                     content: ti.template(),
                                     frame: ti.template(),
                                     cam_pos: tm.vec3,
                                     look_at: tm.vec3,
                                     field_of_vision: float):
    
    look_dir = look_at-cam_pos
    d = vnorm(look_dir)
    look_dir /= d
    
    fw, fh = frame.shape
    r0 = max(fh, fw)
    rx, ry = fw/r0, fh/r0
    
    fovx, fovy = field_of_vision*rx, field_of_vision*ry
    
    xmin = d*tm.tan(fovx/2)
    ymin = d*tm.tan(fovy/2)
    
    M = get_matrix_of_rotation(look_dir)
    iM = get_matrix_of_inverse_rotation(look_dir)
    
    dx = (model_extents.xmax - model_extents.xmin)/model.shape[0]
    dy = (model_extents.ymax - model_extents.ymin)/model.shape[1]
    dz = (model_extents.zmax - model_extents.zmin)/model.shape[2]
    
    w, h = frame.shape
    dX = 2*xmin/w
    dY = 2*ymin/h
    
    for i, j in ti.ndrange(*frame.shape):
        X = tm.vec2([-xmin + i*dX,
                     -ymin + j*dY])
        v = from_cam(X, d, iM)
        
        if ray_box_intersection(cam_pos, v, min(dx, dy, dz), model_extents):
            pos = ray_cast(cam_pos, v, min(dx, dy, dz),
                                parts, parts_n, model_extents)
            X, Y = to_cam(pos, cam_pos, look_dir, d, M)
            x, y, z = pos
            
            ix = int((x-model_extents.xmin)/dx)
            iy = int((y-model_extents.ymin)/dy)
            iz = int((z-model_extents.zmin)/dz)
            
            if (ix >= model.shape[0]-1 or ix < 1 or
                iy >= model.shape[1]-1 or iy < 1 or
                iz >= model.shape[2]-1 or iz < 1):
                frame[i, j] = 0
            else:
                n = normal(ix, iy, iz, model)
                
                frame[i, j] = homogenious_lighting(n, pos, cam_pos)*content[ix, iy, iz]
        else:
            frame[i, j] = 0