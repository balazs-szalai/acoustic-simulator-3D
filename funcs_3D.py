# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:03:13 2024

@author: balazs
"""

import cv2
import numpy as np
import taichi as ti

@ti.func 
def cp_temp(temp: ti.template(), 
            state: ti.template()):
    w, h, d = temp.P.shape
    for i, j, k in ti.ndrange(w, h, d):
        state.vx[i, j, k] = temp.vx[i, j, k]
        state.vy[i, j, k] = temp.vy[i, j, k]
        state.vz[i, j, k] = temp.vz[i, j, k]
        state.P[i, j, k] = temp.P[i, j, k]

@ti.kernel 
def E_ti(state: ti.template(), c0: float, ds: float) -> float:
    T = ti.cast(0, float)
    V = ti.cast(0, float)
    
    w, h, d = state.P.shape
    
    for i, j, k in ti.ndrange(w, h, d):
        i += 1
        j += 1
        k += 1
        
        T += 0.5*state.rho[i, j, k]*(state.vx[i, j, k]**2 + 
                                     state.vy[i, j, k]**2 +
                                     state.vz[i, j, k]**2)*ds**3
        V += 0.5*state.P[i, j, k]**2/(state.rho[i, j, k]*state.cr[i, j, k]**2*c0**2)*ds**3
    return T + V

def E(*index):
    state, c0, ds = index[-3:]
    index = index[:-4]
    if isinstance(index, type(None)):
        P_in = state.P.to_numpy()
        vx_in = state.vx.to_numpy()
        vy_in = state.vy.to_numpy()
        vz_in = state.vy.to_numpy()
        rho_in = state.rho.to_numpy()
        cr_in = state.cr.to_numpy()
        
        T = np.sum(0.5*rho_in*(vx_in**2 + vy_in**2 + vz_in**2)*ds**3)
        V = np.sum(0.5*P_in**2/(rho_in*cr_in**2*c0**2)*ds**3)
        return T + V
    P_in = state.P.to_numpy()[*index]
    vx_in = state.vx.to_numpy()[*index]
    vy_in = state.vy.to_numpy()[*index]
    rho_in = state.rho.to_numpy()[*index]
    cr_in = state.cr.to_numpy()[*index]
    
    T = np.sum(0.5*rho_in*(vx_in**2 + vy_in**2 + vz_in**2)*ds**3)
    V = np.sum(0.5*P_in**2/(rho_in*cr_in**2*c0**2)*ds**3)
    return T + V

@ti.kernel 
def detect(state: ti.template(), 
           detector: ti.template()) -> float:
    P_out = ti.cast(0, float)
    n = 0
    
    w = detector.ixmax - detector.ixmin
    h = detector.iymax - detector.iymin
    d = detector.izmax - detector.izmin
    
    for i, j, k in ti.ndrange(w, h, d):
        i += detector.ixmin
        j += detector.iymin
        k += detector.izmin
        
        P_out += state.P[i, j, k]
        n += 1
    return P_out/n

@ti.kernel 
def detect_in_space(state: ti.template(), 
                    detector: ti.template(),
                    detected: ti.types.ndarray(ti.f64, ndim = 3)):
    w = detector.ixmax - detector.ixmin
    h = detector.iymax - detector.iymin
    d = detector.izmax - detector.izmin
    
    for i, j, k in ti.ndrange(w, h, d):        
        detected[i, j, k] = state.P[i + detector.ixmin, j + detector.iymin, k + detector.izmin]

SFs9p_np = np.array([1/256, -1/32, 7/64, -7/32, 35/128, -7/32, 7/64, -1/32, 1/256])
SFs9p3D_np = np.einsum('i,j,k->ijk', SFs9p_np, SFs9p_np, SFs9p_np)
SFs9p = ti.field(float, shape = (9, 9, 9))
SFs9p.from_numpy(SFs9p3D_np)

SFs11p_np = np.array([-1/1024, 5/512, -45/1024, 15/128, -105/512, 63/256, -105/512, 15/128, -45/1024, 5/512, -1/1024])
SFs11p3D_np = np.einsum('i,j,k->ijk', SFs11p_np, SFs11p_np, SFs11p_np)
SFs11p = ti.field(float, shape = (11, 11, 11))
SFs11p.from_numpy(SFs11p3D_np)

SFs13p_np = np.array([1/4096, -3/1024, 33/2048, -55/1024, 495/4096, -99/512, 231/1024, -99/512, 495/4096, -55/1024, 33/2048, -3/1024, 1/4096])
SFs13p3D_np = np.einsum('i,j,k->ijk', SFs13p_np, SFs13p_np, SFs13p_np)
SFs13p = ti.field(float, shape = (13, 13, 13))
SFs13p.from_numpy(SFs13p3D_np)

@ti.func 
def stencil_filter(s_in: ti.template(), 
                   kernel: ti.template(), 
                   s_out: ti.template()):
    kw, kh, kd = kernel.shape
    padx = int((kw-1)/2)
    pady = int((kh-1)/2)
    padz = int((kd-1)/2)
    w, h, d = s_in.shape
    for x, y, z in ti.ndrange(w-2*padx-1, h-2*pady, d-2*padz):
        y += pady
        x += padx
        z += padz
                
        s_out[x, y, z] = 0
        for kx, ky, kz in ti.ndrange(kw, kh, kd):
            s_out[x, y, z] += s_in[x + kx - padx, y + ky - pady, z + kz - padz] * kernel[kx, ky, kz]
    
    for i, j, k in ti.ndrange(w-2*padx, h-2*pady, d-2*padz):
        i += padx
        j += pady
        k += padz
        
        s_in[i, j, k] -= s_out[i, j, k]

@ti.func 
def generate(t: float, source: ti.template(), temp: ti.template(), A0: float, f0: float, phi: float):
    A = A0*ti.sin(2*np.pi*t*f0 - phi)
    
    w = source.ixmax - source.ixmin
    h = source.iymax - source.iymin
    d = source.izmax - source.izmin
    
    for i, j, k in ti.ndrange(w, h, d):
        i += source.ixmin
        j += source.iymin
        k += source.izmin
        
        temp.P[i, j, k] = A

@ti.func 
def structure(model: ti.template(), temp: ti.template()):
    w, h, d = model.shape
    for i, j, k in ti.ndrange(w, h, d):
        if not model[i, j, k]:
            temp.P[i, j, k] = 0

@ti.func 
def div_v(state: ti.template(), 
          i: int, 
          j: int,
          k: int):
    return ((state.vx[i, j, k] - state.vx[i-1, j, k])
          + (state.vy[i, j, k] - state.vy[i, j-1, k])
          + (state.vz[i, j, k] - state.vz[i, j, k-1]))

@ti.func 
def grad_P_x(state: ti.template(),
             i: int, 
             j: int,
             k: int):
    return state.P[i+1, j, k] - state.P[i, j, k]

@ti.func 
def grad_P_y(state: ti.template(),
             i: int, 
             j: int,
             k: int):
    return state.P[i, j+1, k] - state.P[i, j, k]

@ti.func 
def grad_P_z(state: ti.template(),
             i: int, 
             j: int,
             k: int):
    return state.P[i, j, k+1] - state.P[i, j, k]
    

@ti.func 
def abc_boundary(temp: ti.template(), state: ti.template(), Sc: float):
    '''
    Implements Mur's absorbing boundary condition

    '''
    w, h, d = state.P.shape
    Sc0 = Sc
    c6 = (Sc - 1)/(Sc + 1)
    for i, j in ti.ndrange(w, h):
        Sc = Sc0*state.cr[i, j, d-1]
        c6 = (Sc - 1)/(Sc + 1)
        temp.P[i, j, d-1] = state.P[i, j, d-2] + c6*(temp.P[i, j, d-2] - state.P[i, j, d-1])
        temp.vx[i, j, d-1] = state.vx[i, j, d-2] + c6*(temp.vx[i, j, d-2] - state.vx[i, j, d-1])
        temp.vy[i, j, d-1] = state.vy[i, j, d-2] + c6*(temp.vy[i, j, d-2] - state.vy[i, j, d-1])
        temp.vz[i, j, d-1] = state.vz[i, j, d-2] + c6*(temp.vz[i, j, d-2] - state.vz[i, j, d-1])
        
        Sc = Sc0*state.cr[i, j, 0]
        c6 = (Sc - 1)/(Sc + 1)
        temp.P[i, j, 0] = state.P[i, j, 1] + c6*(temp.P[i, j, 1] - state.P[i, j, 0])
        temp.vx[i, j, 0] = state.vx[i, j, 1] + c6*(temp.vx[i, j, 1] - state.vx[i, j, 0])
        temp.vy[i, j, 0] = state.vy[i, j, 1] + c6*(temp.vy[i, j, 1] - state.vy[i, j, 0])
        temp.vz[i, j, 0] = state.vz[i, j, 1] + c6*(temp.vz[i, j, 1] - state.vz[i, j, 0])
    
    for i, j in ti.ndrange(w, d):
        Sc = Sc0*state.cr[i, h-1, j]
        c6 = (Sc - 1)/(Sc + 1)
        temp.P[i, h-1, j] = state.P[i, h-2, j] + c6*(temp.P[i, h-2, j] - state.P[i, h-1, j])
        temp.vx[i, h-1, j] = state.vx[i, h-2, j] + c6*(temp.vx[i, h-2, j] - state.vx[i, h-1, j])
        temp.vy[i, h-1, j] = state.vy[i, h-2, j] + c6*(temp.vy[i, h-2, j] - state.vy[i, h-1, j])
        temp.vz[i, h-1, j] = state.vz[i, h-2, j] + c6*(temp.vz[i, h-2, j] - state.vz[i, h-1, j])
        
        Sc = Sc0*state.cr[i, 0, j]
        c6 = (Sc - 1)/(Sc + 1)
        temp.P[i, 0, j] = state.P[i, 1, j] + c6*(temp.P[i, 1, j] - state.P[i, 0, j])
        temp.vx[i, 0, j] = state.vx[i, 1, j] + c6*(temp.vx[i, 1, j] - state.vx[i, 0, j])
        temp.vy[i, 0, j] = state.vy[i, 1, j] + c6*(temp.vy[i, 1, j] - state.vy[i, 0, j])
        temp.vz[i, 0, j] = state.vz[i, 1, j] + c6*(temp.vz[i, 1, j] - state.vz[i, 0, j])
    
    for i, j in ti.ndrange(h, d):
        Sc = Sc0*state.cr[w-1, i, j]
        c6 = (Sc - 1)/(Sc + 1)
        temp.P[w-1, i, j] = state.P[w-2, i, j] + c6*(temp.P[w-2, i, j] - state.P[w-1, i, j])
        temp.vx[w-1, i, j] = state.vx[w-2, i, j] + c6*(temp.vx[w-2, i, j] - state.vx[w-1, i, j])
        temp.vy[w-1, i, j] = state.vy[w-2, i, j] + c6*(temp.vy[w-2, i, j] - state.vy[w-1, i, j])
        temp.vz[w-1, i, j] = state.vz[w-2, i, j] + c6*(temp.vz[w-2, i, j] - state.vz[w-1, i, j])
        
        Sc = Sc0*state.cr[0, i, j]
        c6 = (Sc - 1)/(Sc + 1)
        temp.P[0, i, j] = state.P[1, i, j] + c6*(temp.P[1, i, j] - state.P[0, i, j])
        temp.vx[0, i, j] = state.vx[1, i, j] + c6*(temp.vx[1, i, j] - state.vx[0, i, j])
        temp.vy[0, i, j] = state.vy[1, i, j] + c6*(temp.vy[1, i, j] - state.vy[0, i, j])
        temp.vz[0, i, j] = state.vz[1, i, j] + c6*(temp.vz[1, i, j] - state.vz[0, i, j])

@ti.func 
def comp_next(state: ti.template(), 
            temp: ti.template(),
            k1: ti.template(), 
            c0: float, 
            Sc: float,
            coef: float):
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1
        
        Cvx = 2*Sc/(c0*(state.rho[i+1, j, k] + state.rho[i, j, k]))
        Cvy = 2*Sc/(c0*(state.rho[i, j+1, k] + state.rho[i, j, k]))
        Cvz = 2*Sc/(c0*(state.rho[i, j, k+1] + state.rho[i, j, k]))
        CP = state.rho[i, j, k]*state.cr[i, j, k]**2*c0*Sc
        
        k1.P[i, j, k]  = - CP * div_v(temp, i, j, k) * coef
        k1.vx[i, j, k] = - Cvx * grad_P_x(temp, i, j, k) * coef
        k1.vy[i, j, k] = - Cvy * grad_P_y(temp, i, j, k) * coef
        k1.vz[i, j, k] = - Cvz * grad_P_z(temp, i, j, k) * coef

@ti.func 
def comp_k1(state: ti.template(),
            k1: ti.template(), 
            c0: float, 
            Sc: float):
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1
        
        Cvx = 2*Sc/(c0*(state.rho[i+1, j, k] + state.rho[i, j, k]))
        Cvy = 2*Sc/(c0*(state.rho[i, j+1, k] + state.rho[i, j, k]))
        Cvz = 2*Sc/(c0*(state.rho[i, j, k+1] + state.rho[i, j, k]))
        CP = state.rho[i, j, k]*state.cr[i, j, k]**2*c0*Sc
        
        k1.P[i, j, k]  = - CP * div_v(state, i, j, k)
        k1.vx[i, j, k] = - Cvx * grad_P_x(state, i, j, k)
        k1.vy[i, j, k] = - Cvy * grad_P_y(state, i, j, k)
        k1.vz[i, j, k] = - Cvz * grad_P_z(state, i, j, k)

@ti.func 
def comp_k2(state: ti.template(),
            k2: ti.template(),
            k1: ti.template(), 
            c0: float, 
            Sc: float):
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1
        
        Cvx = 2*Sc/(c0*(state.rho[i+1, j, k] + state.rho[i, j, k]))
        Cvy = 2*Sc/(c0*(state.rho[i, j+1, k] + state.rho[i, j, k]))
        Cvz = 2*Sc/(c0*(state.rho[i, j, k+1] + state.rho[i, j, k]))
        CP = state.rho[i, j, k]*state.cr[i, j, k]**2*c0*Sc
        
        k2.P[i, j, k]  = - CP * div_v(k1, i, j, k)/2
        k2.vx[i, j, k] = - Cvx * grad_P_x(k1, i, j, k)/2
        k2.vy[i, j, k] = - Cvy * grad_P_y(k1, i, j, k)/2
        k2.vz[i, j, k] = - Cvz * grad_P_z(k1, i, j, k)/2

@ti.func 
def comp_k3(state: ti.template(),
            k3: ti.template(),
            k2: ti.template(),
            k1: ti.template(),
            c0: float, 
            Sc: float):
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1
        
        Cvx = 2*Sc/(c0*(state.rho[i+1, j, k] + state.rho[i, j, k]))
        Cvy = 2*Sc/(c0*(state.rho[i, j+1, k] + state.rho[i, j, k]))
        Cvz = 2*Sc/(c0*(state.rho[i, j, k+1] + state.rho[i, j, k]))
        CP = state.rho[i, j, k]*state.cr[i, j, k]**2*c0*Sc
        
        k3.P[i, j, k]  = - CP * div_v(k2, i, j, k)/2
        k3.vx[i, j, k] = - Cvx * grad_P_x(k2, i, j, k)/2
        k3.vy[i, j, k] = - Cvy * grad_P_y(k2, i, j, k)/2
        k3.vz[i, j, k] = - Cvz * grad_P_z(k2, i, j, k)/2

@ti.func 
def comp_k4(state: ti.template(),
            k4: ti.template(),
            k3: ti.template(),
            k1: ti.template(),
            c0: float, 
            Sc: float):
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1
        
        Cvx = 2*Sc/(c0*(state.rho[i+1, j, k] + state.rho[i, j, k]))
        Cvy = 2*Sc/(c0*(state.rho[i, j+1, k] + state.rho[i, j, k]))
        Cvz = 2*Sc/(c0*(state.rho[i, j, k+1] + state.rho[i, j, k]))
        CP = state.rho[i, j, k]*state.cr[i, j, k]**2*c0*Sc
        
        k4.P[i, j, k]  = - CP * div_v(k3, i, j, k)
        k4.vx[i, j, k] = - Cvx * grad_P_x(k3, i, j, k)
        k4.vy[i, j, k] = - Cvy * grad_P_y(k3, i, j, k)
        k4.vz[i, j, k] = - Cvz * grad_P_z(k3, i, j, k)

@ti.func 
def RK4(state: ti.template(),
        temp: ti.template(),
        k1: ti.template(), 
        k2: ti.template(), 
        k3: ti.template(), 
        k4: ti.template(), 
        c0: float, 
        Sc: float):
    comp_k1(state, k1, c0, Sc)
    
    comp_k2(state, k2, k1, c0, Sc)
    
    comp_k3(state, k3, k2, k1, c0, Sc)
    
    comp_k4(state, k4, k3, k1, c0, Sc)
    
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1  
        
        temp.P[i, j, k] = state.P[i, j, k] + (k1.P[i, j, k] + 2*k2.P[i, j, k] + 2*k3.P[i, j, k] + k4.P[i, j, k])/6
        temp.vx[i, j, k] = state.vx[i, j, k] + (k1.vx[i, j, k] + 2*k2.vx[i, j, k] + 2*k3.vx[i, j, k] + k4.vx[i, j, k])/6
        temp.vy[i, j, k] = state.vy[i, j, k] + (k1.vy[i, j, k] + 2*k2.vy[i, j, k] + 2*k3.vy[i, j, k] + k4.vy[i, j, k])/6
        temp.vz[i, j, k] = state.vz[i, j, k] + (k1.vz[i, j, k] + 2*k2.vz[i, j, k] + 2*k3.vz[i, j, k] + k4.vz[i, j, k])/6

@ti.func 
def add_states(state1: ti.template(),
               state2: ti.template()):
    w, h, d = state1.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1 
        
        state1.P[i, j, k] += state2.P[i, j, k]
        state1.vx[i, j, k] += state2.vx[i, j, k]
        state1.vy[i, j, k] += state2.vy[i, j, k]
        state1.vz[i, j, k] += state2.vz[i, j, k]

@ti.func 
def mul_state(state1: ti.template(),
              coef: float):
    w, h, d = state1.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1
        
        state1.P[i, j, k] *= coef
        state1.vx[i, j, k] *= coef
        state1.vy[i, j, k] *= coef
        state1.vz[i, j, k] *= coef

@ti.func
def higher_order_Taylor(state: ti.template(),
                        temp: ti.template(),
                        temp1: ti.template(),
                        temp2: ti.template(),
                        c0: float,
                        Sc: float):
    
    comp_next(state, state, temp1, c0, Sc, 1)
    comp_next(state, temp1, temp2, c0, Sc, 1/2)
    add_states(temp, temp1)
    add_states(temp, temp2)
    
    comp_next(state, temp2, temp1, c0, Sc, 1/3)
    comp_next(state, temp1, temp2, c0, Sc, 1/4)
    add_states(temp, temp1)
    add_states(temp, temp2)
    
    

@ti.func 
def Euler(state: ti.template(),
        temp: ti.template(),
        k1: ti.template(),
        c0: float, 
        Sc: float):
    comp_k1(state, k1, c0, Sc)
    
    w, h, d = state.P.shape
    for i, j, k in ti.ndrange(w-2, h-2, d-2):
        i += 1
        j += 1
        k += 1  
        
        temp.P[i, j, k] = state.P[i, j, k] + k1.P[i, j, k]
        temp.vx[i, j, k] = state.vx[i, j, k] + k1.vx[i, j, k]
        temp.vy[i, j, k] = state.vy[i, j, k] + k1.vy[i, j, k]
        temp.vz[i, j, k] = state.vz[i, j, k] + k1.vz[i, j, k]

@ti.kernel 
def step(state: ti.template(),
         temp: ti.template(),
         k1: ti.template(),
         k2: ti.template(),
         k3: ti.template(),
         k4: ti.template(),
         source: ti.template(),
         A0: float,
         f0: float,
         c0: float,
         Sc: float,
         t: float,
         phi: float):
    # stencil_filter(state, SFs13p, temp, Pa)
    
    # RK4(state, 
    #     temp, 
    #     k1, 
    #     k2, 
    #     k3, 
    #     k4, 
    #     c0, 
    #     Sc)
    # Euler(state, temp, k1, c0, Sc)
    higher_order_Taylor(state, temp, k1, k2, c0, Sc)
        
    generate(t, 
             source, 
             temp, 
             A0, 
             f0,
             phi)
    abc_boundary(temp, 
                 state, 
                 Sc)
    cp_temp(temp, 
            state)

@ti.kernel 
def apply_filter(arr: ti.template(),
                 kernel: ti.template(),
                 temp: ti.template()):
    stencil_filter(arr, kernel, temp)
    w, h, d = arr.shape
    
    for i, j, k in ti.ndrange(w, h, d):
        temp[i, j, k] = arr[i, j, k]