# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 23:58:43 2019

@author: ohman
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#均一性の評価
def check_uniformity(u_int, x, y):
    u_int = u_int / np.max(u_int)

    uni = []
    for i , j in zip(x,y):
        uni.append(u_int[i][j])

    maxi = max(uni)
    mini = min(uni)
    uniformity = 1 - (maxi-mini)/(maxi+mini)
    print("均一性:", uniformity)
    return uniformity


def hologram(phase):
    phase = np.where(phase<0, phase+2*np.pi, phase)
    p_max = np.max(phase)
    p_min = np.min(phase)
    holo = ((phase - p_min)/(p_max- p_min)) * 255
    holo = holo.astype("uint8")
    return holo


def reconstruct(u_int):
    u_max = np.max(u_int)
    u_min = np.min(u_int)
    rec = ((u_int - u_min)/(u_max - u_min)) * 255
    rec = rec.astype("uint8")
    return rec
    
def main():
    target = cv2.imread("img/target.bmp",0)
    cv2.imshow("target",target)
    cv2.waitKey(0)
    
    height, width = target.shape[:2]
    target[target>150] = 255
    
    #スポットの場所
    x,y = np.where(target == 255)
    
    target = target / 255
    laser = 1
    phase = np.random.rand(height, width)
    u = np.empty_like(target, dtype="complex")
    
    iteration = 30
    uniformity = []
    
    for num in range(iteration):
        u.real = laser * np.cos(phase)
        u.imag = laser * np.sin(phase)
        
        #-------レンズ---------
        u = np.fft.fft2(u)
        u = np.fft.fftshift(u)
        #-------レンズ---------
        
        u_abs = np.abs(u)
        u_int = u_abs ** 2
        
        uniformity.append(check_uniformity(u_int, x, y))
        
        phase = np.angle(u)
        
        u.real = target * np.cos(phase)
        u.imag = target * np.sin(phase)
        
        #-----レンズ---------
        u = np.fft.ifftshift(u)
        u = np.fft.ifft2(u)
        #-------レンズ---------
        
        phase = np.angle(u)
        
    
    holo_name = "hologram"
    rec_name = "reconstruction"
    
    holo = hologram(phase)
    cv2.imwrite("img/{}.bmp".format(holo_name), holo)
    cv2.imshow("Hologram", holo)
    cv2.waitKey(0)
    
    rec = reconstruct(u_int)
    cv2.imwrite("img/{}.bmp".format(rec_name), rec)
    cv2.imshow("Reconstruction", rec)
    cv2.waitKey(0)
    
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1,iteration+1),uniformity)
    plt.xlabel("Iteration")
    plt.ylabel("Uniformity")
    plt.ylim(0,1)
    
if __name__ == "__main__":
    main()
    
