# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 00:32:16 2019

@author: ohman
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#重み付け
def weight(w_u, target, before_u, norm_int):
    w_u[target==1] = ((target[target==1] / norm_int[target==1]) ** 0.5) * before_u[target==1] 
    return w_u

#均一性の評価
def check_uniformity(u_int, target):
    u_int = u_int / np.max(u_int)

    maxi = np.max(u_int[target==1])
    mini = np.min(u_int[target==1])
    uniformity = 1 - (maxi-mini)/(maxi+mini)
    print("均一性:", uniformity)
    return uniformity

def normalization(origin):
    maxi = np.max(origin)
    mini = np.min(origin)
    norm = ((origin - mini) / (maxi - mini))
    return norm


def hologram(phase):
    phase = np.where(phase<0, phase+2*np.pi, phase)
    holo  = normalization(phase) * 255
    holo = holo.astype("uint8")
    return holo


def reconstruct(norm_int):
    rec = norm_int * 255
    rec = rec.astype("uint8")
    return rec
    
def main():
    target = cv2.imread("img/target.bmp",0)
    cv2.imshow("target",target)
    cv2.waitKey(0)
    
    height, width = target.shape[:2]
    target[target>150] = 255
    
    
    target = target / 255
    laser = 1
    phase = np.random.rand(height, width)
    u = np.empty_like(target, dtype="complex")
    before_u = target
    w_u = np.empty_like(target)
    
    iteration = 30
    uniformity = []
    
    for num in range(iteration):
        u.real = laser * np.cos(phase)
        u.imag = laser * np.sin(phase)
        
        #-------レンズ---------
        u = np.fft.fft2(u)
        u = np.fft.fftshift(u)
        #-------レンズ---------
    
        u_int = np.abs(u) ** 2
        norm_int = normalization(u_int)
        
        uniformity.append(check_uniformity(u_int, target))
        
        phase = np.angle(u)
        
        w_u = weight(w_u, target, before_u, norm_int)
        w_u = normalization(w_u)
        before_u = w_u
        
        u.real = w_u * np.cos(phase)
        u.imag = w_u * np.sin(phase)
        
        #-----レンズ---------
        u = np.fft.ifftshift(u)
        u = np.fft.ifft2(u)
        #-------レンズ---------
        
        phase = np.angle(u)
        
        
    #効率性
    efficiency = np.sum(norm_int[target==1]) / np.sum(target[target==1])
    print("効率性 : ", efficiency)
    
    holo_name = "weighted_hologram"
    rec_name = "weighted_reconstruction"
        
    holo = hologram(phase)
    cv2.imwrite("img/{}.bmp".format(holo_name), holo)
    cv2.imshow("Weighted Hologram", holo)
    cv2.waitKey(0)
    
    rec = reconstruct(norm_int)
    cv2.imwrite("img/{}.bmp".format(rec_name), rec)
    cv2.imshow("Weighted Reconstruction", rec)
    cv2.waitKey(0)
    
    plt.figure(figsize=(10,8))
    plt.plot(np.arange(1,iteration+1),uniformity)
    plt.xlabel("Iteration")
    plt.ylabel("Uniformity")
    plt.ylim(0,1)

if __name__ == "__main__":
    main()
    