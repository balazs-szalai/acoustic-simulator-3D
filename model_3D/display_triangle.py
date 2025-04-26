# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:57:24 2024

@author: balazs
"""

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
ti.init(ti.gpu)

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


def vnorm(vi):
    v = np.array([*vi])
    return np.sqrt(np.sum(v**2))

@ti.func 
def vnorm2(v):
    return tm.sqrt(v[0]**2 + v[1]**2)

@ti.func 
def vnorm3(v):
    return tm.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def scalar_prod(u, v):
    return np.sum(u*v)

@ti.func 
def scalar_prod2(u, v):
    return u[0]*v[0] + u[1]*v[1]

@ti.func 
def scalar_prod3(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def sgn(x):
    return (x>0)*2 - 1

@ti.func
def in_triangle(x, a, b, c):
    u = b-a
    v = c-a
    w = x-a
    
    ul = vnorm2(u)
    vl = vnorm2(v)
    wl = vnorm2(w)
    
    p1 = scalar_prod2(u, v)/(ul*vl)
    p2 = scalar_prod2(w, u)/(wl*ul)
    p3 = scalar_prod2(w, v)/(wl*vl)
    
    k1 = c-b
    k2 = tm.vec2([k1[1], -k1[0]])
    
    k2 /= vnorm2(k2)
    
    d = abs(scalar_prod2(k2, u))
    
    ret = False
    if abs(scalar_prod2(w, k2)) < d and p1 < p2 and p1 < p3:
        ret = True
    return ret

@ti.func 
def point_to_ind(X: tm.vec2,
                 xmin: float,
                 ymin: float,
                 dx: float,
                 dy: float):
    x, y = X
    x += xmin
    y += ymin
    
    ix = int(x/dx)
    iy = int(y/dy)
    
    return tm.ivec2(ix, iy)



eps = 1e-6
@ti.func
def render_flat_triangle(a: tm.vec2, 
                         b: tm.vec2, 
                         c: tm.vec2, 
                         I: float,
                         da: float,
                         db: float,
                         dc: float,
                         xmin: float,
                         ymin: float,
                         frame_ti: ti.template(),
                         depth_buffer: ti.template()):
    w, h = frame_ti.shape
    dx = 2*xmin/w
    dy = 2*ymin/h
    # for i, j in ti.ndrange(w, h):
    #     if in_triangle(tm.vec2([-xmin + i*dx, -ymin + j*dy]),
    #                                                 a, b, c):
    #         frame_ti[i, j] = I
    min_point = tm.min(a, b, c)
    max_point = tm.max(a, b, c)
    
    ixmin, iymin = point_to_ind(min_point, xmin, ymin, dx, dy)
    ixmax, iymax = point_to_ind(max_point, xmin, ymin, dx, dy)
    
    ab = b-a
    ac = c-a
    
    m = tm.mat2([[ab[0], ac[0]],
                 [ab[1]], ac[1]])
    
    im = tm.inverse(m)
    
    for i, j in ti.ndrange(ixmax-ixmin, iymax-iymin): 
        i += ixmin
        j += iymin
        
        X = tm.vec2([-xmin + i*dx, -ymin + j*dy])
        if in_triangle(X, a, b, c):
            
            b_comp, c_comp = im@(X-a)
            
            d = da + b_comp*(db-da) + c_comp*(dc-da)
            ti.atomic_min(depth_buffer[i, j], d)
        
    for i, j in ti.ndrange(ixmax-ixmin, iymax-iymin): 
        i += ixmin
        j += iymin
        
        X = tm.vec2([-xmin + i*dx, -ymin + j*dy])
        if in_triangle(X, a, b, c):
            b_comp, c_comp = im@(X-a)
            d = da + b_comp*(db-da) + c_comp*(dc-da)
            if abs(d - depth_buffer[i, j]) < eps:
                frame_ti[i, j] = I
    
    
    # print('--------------')
    # X = min_point
    # count = 0
    # while X[1] < max_point[1] and count < 100000:
    #     while in_triangle(X, a, b, c) and count < 100000:
    #         print(X)
    #         X[0] -= dx
    #         count += 1
    #     while not in_triangle(X, a, b, c) and count < 100000:
    #         print(X)
    #         X[0] += dx
    #         count += 1
    #     x0 = X[0]
    #     # print(in_triangle(X, a, b, c))
        
    #     while in_triangle(X, a, b, c) and count < 100000:
    #         print(X)
    #         ix, iy = point_to_ind(X, xmin, ymin, dx, dy)
    #         if ix >= 0 and ix < w and iy >= 0 and iy < h:
    #             frame_ti[ix, iy] = I
    #         X[0] += dx
    #         count += 1
    #     X[1] += dy
    #     X[0] = x0
        
    #     count += 1

@ti.func 
def proj(v, n):
    return v - n*scalar_prod3(n, v)

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
def render_triangle(a: tm.vec3,
                    b: tm.vec3,
                    c: tm.vec3,
                    n: tm.vec3,
                    cam_pos: tm.vec3,
                    look_at: tm.vec3,
                    fovx: float,
                    fovy: float,
                    frame_ti: ti.template(),
                    depth_buffer: ti.template()):
    # u = b-a
    # v = c-a
    
    # n = vec_prod(u, v) 
    # n /= vnorm3(n)
    
    look_dir = look_at-cam_pos
    d = vnorm3(look_dir)
    look_dir /= d
    
    I = abs(scalar_prod3(look_dir, n))
    
    ac = a-cam_pos
    bc = b-cam_pos
    cc = c-cam_pos
    
    da = vnorm3(ac)
    db = vnorm3(bc)
    dc = vnorm3(cc)
    
    M = get_matrix_of_rotation(look_dir)
    
    a_rot = M@ac
    b_rot = M@bc
    c_rot = M@cc
    
    ln = tm.vec3([0, 0, 1])
    
    ap = proj(a_rot, ln)
    bp = proj(b_rot, ln)
    cp = proj(c_rot, ln)
    
    ap2 = tm.vec2([ap[0], ap[1]])
    bp2 = tm.vec2([bp[0], bp[1]])
    cp2 = tm.vec2([cp[0], cp[1]])
    
    xmin = d*tm.tan(fovx/2)
    ymin = d*tm.tan(fovy/2)
    
    render_flat_triangle(ap2, bp2, cp2, I, da, db, dc, xmin, ymin, frame_ti, depth_buffer)

@ti.kernel 
def render_solid(triangles: ti.template(),
                 cam_pos: tm.vec3,
                 look_at: tm.vec3,
                 fovx: float,
                 fovy: float,
                 frame_ti: ti.template(),
                 depth_buffer: ti.template()):
    l, w = triangles.shape
    # print(triangles.shape)
    for i in ti.ndrange(l):
        render_triangle(triangles[i, 0],
                        triangles[i, 1],
                        triangles[i, 2],
                        triangles[i, 3],
                        cam_pos,
                        look_at,
                        fovx,
                        fovy,
                        frame_ti,
                        depth_buffer)

class Triangle:
    def __init__(self, vertices, frame_shape, look_at, cam_pos, fovx, fovy):
        self.vertices = vertices
        self.a = vertices[0]
        self.b = vertices[1]
        self.c = vertices[2]
        
        u = self.b - self.a 
        v = self.c - self.a
        
        n = vec_prod_f(u, v)
        
        self.n = n/vnorm(n)
        
        self.look_at = look_at
        self.cam_pos = cam_pos
        self.frame = np.zeros(frame_shape)
        
        self.frame_ti = ti.field(float, shape=frame_shape)
        
        # extents_ti = ti.field(float, shape=(2, 2))
        # extents_ti[0, 0] = extents[0][0]
        # extents_ti[0, 1] = extents[0][1]
        # extents_ti[1, 0] = extents[1][0]
        # extents_ti[1, 1] = extents[1][1]
        
        # self.extents_ti = extents_ti
        
        self.fovx = fovx
        self.fovy = fovy
        self.xmin = vnorm(cam_pos-look_at)*np.tan(fovx/2)
        self.ymin = vnorm(cam_pos-look_at)*np.tan(fovy/2)
     
    def render(self):
        render_triangle(self.a,
                        self.b,
                        self.c,
                        self.n,
                        self.cam_pos,
                        self.look_at,
                        self.fovx,
                        self.fovy,
                        self.frame_ti)
        self.frame = self.frame_ti.to_numpy()
    
    def show(self, render = True):
        if render:
            self.frame_ti.fill(0)
            self.render()
        plt.imshow(self.frame, aspect = 'auto', extent=[-self.xmin, self.xmin,
                                                        -self.ymin, self.xmin],
                   origin='upper')
        plt.pause(0.01)


class Solid:
    def __init__(self, frame_shape, look_at, cam_pos, fovx, fovy, triangles = None):
        self.triangles = triangles
        
        self.look_at = look_at
        self.cam_pos = cam_pos
        self.frame = np.zeros(frame_shape)
        
        self.frame_ti = ti.field(float, shape=frame_shape)
        self.depth_buffer = ti.field(float, shape = frame_shape)
        self.depth_buffer.fill(3e38)
        
        self.fovx = fovx
        self.fovy = fovy
        self.xmin = vnorm(cam_pos-look_at)*np.tan(fovx/2)
        self.ymin = vnorm(cam_pos-look_at)*np.tan(fovy/2)
     
    def render(self):
        self.frame_ti.fill(0)
        self.depth_buffer.fill(3e38)
        render_solid(self.triangles,
                     self.cam_pos,
                     self.look_at,
                     self.fovx,
                     self.fovy,
                     self.frame_ti,
                     self.depth_buffer)
        self.frame = self.frame_ti.to_numpy()
    
    def show(self, render = True):
        if render:
            self.render()
        plt.imshow(self.frame, aspect = 'auto', extent=[-self.xmin, self.xmin,
                                                        -self.ymin, self.xmin],
                   origin='upper')
        plt.pause(0.01)
    
    def read_stl(self, file):
        data = []
        with open(file, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                
                if line.startswith('facet normal'):
                    one_trg = []
                    line = line[13:].strip()
                    n = [float(i) for i in line.split(' ')]
                    line = f.readline()
                    if line.startswith('outer loop'):
                        while not line == '':
                            line = f.readline()
                            if line.startswith('endloop'):
                                break
                            line = line[7:].strip()
                            vec = [float(i) for i in line.split(' ')]
                            one_trg.append(vec)
                    one_trg.append(n)
                    data.append(one_trg)
        data = np.array(data)
        if not self.triangles:
            self.triangles = ti.field(tm.vec3, shape=data.shape[:-1])
            self.triangles.from_numpy(data)
        else:
            print('the object is already initialised with object data')
            
                

# ext = ti.field(float, shape=(2, 2))
# ext[0, 0] = -2
# ext[0, 1] = -2
# ext[1, 0] = 2
# ext[1, 1] = 2

# frame_ti = ti.field(float, shape=(1000, 1000))

# render_flat_triangle(tm.vec2([1,1]),
#                      tm.vec2([1.2,1.5]),
#                      tm.vec2([1.5,1.2]), 1,
#                      frame_ti, ext)

# plt.imshow(frame_ti.to_numpy())

# trg = Triangle([tm.vec3(0, 1, 1),
#                 tm.vec3(0, 1, 0),
#                 tm.vec3(0, 0, 0)], 
#                (1000, 1000), 
#                tm.vec3([0, 0, 0]), 
#                tm.vec3([2, 0, 0]), 
#                1, 1)

triangles = ti.field(tm.vec3, shape = (1, 4))

triangles.from_numpy(np.array([(0, 0, 1),
                               (0, 1, 0),
                               (0, 0, 0),
                               [1, 0, 0]]).reshape(1, 4, 3))#np.random.random((1, 4, 3)))

sld = Solid((800, 600), 
            tm.vec3([0.5, 0.5, 0.50]), 
            tm.vec3([2, 0, 0.50]), 
            0.8, 0.6)
sld.read_stl('C:/Users/balaz/Desktop/MFF_UK/2_rocnik_LS/pocitacova_fyzika/acoustic_resonator.stl')