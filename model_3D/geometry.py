# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:37:21 2024

@author: balazs
"""

import taichi as ti
import taichi.math as tm

@ti.func 
def is_boundary(i: int, j: int, k: int, model: ti.template()):
    ret = False
    if model[i, j, k]:
        inside = ti.cast(1, ti.u1)
        for l, m, p in ti.ndrange(3, 3, 3):
            inside *= model[i-1+l, j-1+m, k-1+p]
        if not inside:
            ret = True
    return ret
    

@ti.func 
def vnorm(v: tm.vec3):
    return tm.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

@ti.func 
def scalar(v: tm.vec3,
           u: tm.vec3):
    return v[0]*u[0] + v[1]*u[1] + v[2]*u[2]

@ti.func
def mirror(v: tm.vec3,
           q: tm.vec3) -> tm.vec3:
    a = q*scalar(q, v)
    return 2*a - v

@ti.func
def in_cylinder(x: float, y: float, z:float, center: tm.vec3, r:float, h:float):
    ret = False
    if (((x-center[0])**2 + (y-center[1])**2) < r**2 and
        (z > center[2]-h/2 and z < center[2]+h/2)):
        ret = True
    return ret

@ti.kernel 
def add_cylinder(model: ti.template(), model_extents: ti.template(),
                 center: tm.vec3, r:float, h:float):
    
    dx = (model_extents.xmax - model_extents.xmin)/model.shape[0]
    dy = (model_extents.ymax - model_extents.ymin)/model.shape[1]
    dz = (model_extents.zmax - model_extents.zmin)/model.shape[2]
    
    for i, j, k in ti.ndrange(*model.shape):
        x = model_extents.xmin + i*dx
        y = model_extents.ymin + j*dy
        z = model_extents.zmin + k*dz
        
        if not model[i, j, k]:
            model[i, j, k] = in_cylinder(x, y, z, center, r, h)

@ti.func
def in_block(x: float, y: float, z:float, center: tm.vec3, w:float, h:float, d:float):
    ret = False
    if ((x > center[0]-w/2 and x < center[0]+w/2) and
        (y > center[1]-d/2 and y < center[1]+d/2) and
        (z > center[2]-h/2 and z < center[2]+h/2)):
        ret = True
    return ret

@ti.kernel 
def add_block(model: ti.template(), model_extents: ti.template(),
                 center: tm.vec3, w:float, h:float, d:float):
    
    dx = (model_extents.xmax - model_extents.xmin)/model.shape[0]
    dy = (model_extents.ymax - model_extents.ymin)/model.shape[1]
    dz = (model_extents.zmax - model_extents.zmin)/model.shape[2]
    
    for i, j, k in ti.ndrange(*model.shape):
        x = model_extents.xmin + i*dx
        y = model_extents.ymin + j*dy
        z = model_extents.zmin + k*dz
        
        if not model[i, j, k]:
            model[i, j, k] = in_block(x, y, z, center, w, h, d)

@ti.func 
def in_model(x: float, y: float, z: float, 
             parts: ti.template(),
             n: int):
    ret = False
    for i in range(n):
        if not ret:
            if parts[i, 0] == 0:
                ret = in_cylinder(x, y, z, tm.vec3(parts[i, 1], parts[i, 2], parts[i, 3]), 
                                   parts[i, 4], parts[i, 5])
            elif parts[i, 0] == 1:
                ret = in_block(x, y, z, tm.vec3(parts[i, 1], parts[i, 2], parts[i, 3]), 
                                parts[i, 4], parts[i, 5], parts[i, 6])
    return ret

@ti.func 
def ray_box_intersection(cam_pos: tm.vec3,
                         v: tm.vec3,
                         ds: float,
                         model_extents: ti.template()):
    
    tminx = (model_extents.xmin - cam_pos[0])/v[0]
    tmaxx = (model_extents.xmax - cam_pos[0])/v[0]
    if tminx > tmaxx:
        tminx, tmaxx = tmaxx, tminx
    
    tminy = (model_extents.ymin - cam_pos[1])/v[1]
    tmaxy = (model_extents.ymax - cam_pos[1])/v[1]
    if tminy > tmaxy:
        tminy, tmaxy = tmaxy, tminy
    
    tminz = (model_extents.zmin - cam_pos[2])/v[2]
    tmaxz = (model_extents.zmax - cam_pos[2])/v[2]
    if tminz > tmaxz:
        tminz, tmaxz = tmaxz, tminz
    
    t_min = max(tminx, tminy, tminz)
    t_max = min(tmaxx, tmaxy, tmaxz)
    
    ret = False
    if t_min < t_max:
        ret = True
    return ret

@ti.func 
def ray_cast(cam_pos: tm.vec3,
        v: tm.vec3,
        ds: float,
        model_parts: ti.template(), n: int,
        model_extents: ti.template()):
    
    tminx = (model_extents.xmin - cam_pos[0])/v[0]
    tmaxx = (model_extents.xmax - cam_pos[0])/v[0]
    if tminx > tmaxx:
        tminx, tmaxx = tmaxx, tminx
    
    tminy = (model_extents.ymin - cam_pos[1])/v[1]
    tmaxy = (model_extents.ymax - cam_pos[1])/v[1]
    if tminy > tmaxy:
        tminy, tmaxy = tmaxy, tminy
    
    tminz = (model_extents.zmin - cam_pos[2])/v[2]
    tmaxz = (model_extents.zmax - cam_pos[2])/v[2]
    if tminz > tmaxz:
        tminz, tmaxz = tmaxz, tminz
    
    t_min = max(tminx, tminy, tminz)
    t_max = min(tmaxx, tmaxy, tmaxz)
    
    u = v*ds
    
    start = int(t_min/ds)
    end = int(t_max/ds)
    
    collision_point = tm.vec3(model_extents.xmin-1,
                              model_extents.ymin-1,
                              model_extents.zmin-1)
    
    last_in = False
    for i in range(start, end):
        x, y, z = cam_pos + u*i
        
        if not last_in:
            if in_model(x, y, z, model_parts, n):
                collision_point = tm.vec3(x, y, z)
                last_in = True
    return collision_point

part_ids = {'cylinder': 0,
            'block': 1}