# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:56:39 2014

@author: sanrobo
"""
import cv2,time
import pymeanshift as pms#meanShift
import numpy as np

#Öncelikle resm yüklenir
img = cv2.imread('path.jpg')
img_=img.copy()
kernel = np.ones((5,5),np.uint8)

#1-golge tespiti yapilir
#elde edilen seg_image resmi sanki yagli boya resim gibi durmakta
start=time.time()
gauss=cv2.GaussianBlur(img,(5,5),0.4,0.1)
#mean shift algoritması ile resimin renk düzeni tek düze hale getiriliyor
#min_density, arttırıldıkca gurultu daha iyi giderilmekte
(seg_image, lbl_image, nmb_regions) = pms.segment(gauss, 
                                                  spatial_radius=2,
                                                  range_radius=2, 
                                                  min_density=20)
#resimdeki gurultu gideriliyor

#kucuk bozuklklar siliniyor
morphology = cv2.morphologyEx(seg_image, cv2.MORPH_OPEN, kernel)
#resim gri formata donusturuluyor
gray = cv2.cvtColor(morphology,cv2.COLOR_BGR2GRAY)
#resim siyah-beyaz formata donusturuluyor
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,2)
#thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,3,2)
            
Negate=cv2.bitwise_not(thresh)
edges = cv2.Canny(Negate,100,200)
edges_cpy=edges.copy()

edges_=cv2.bitwise_not(edges)

img=img_.copy()
img_1=cv2.bitwise_and(img,img,mask=thresh)#bg
img=img_.copy()
img_2=cv2.bitwise_and(img,img,mask=Negate)#fg
img=img_.copy()
img_3=cv2.bitwise_and(img,img,mask=edges_)#fg

#2-golgenin sinirlari belirleniyor

bg = cv2.GaussianBlur(img_1, (5,5),0.4)
fg = cv2.GaussianBlur(img_2, (5,5),0.4)
bg_fg= cv2.GaussianBlur(img_3, (5,5),0.4)

"""(b_fg,g_fg,r_fg)=cv2.split(fg)
sayac_fg=(np.size(fg,axis=1)*np.size(fg,axis=0)- np.size(fg[fg==0],axis=0))
b_fg_mean=np.sum(b_fg)/sayac_fg
g_fg_mean=np.sum(g_fg)/sayac_fg
r_fg_mean=np.sum(r_fg)/sayac_fg

(b_bg,g_bg,r_bg)=cv2.split(bg)
sayac_bg=(np.size(bg,axis=1)*np.size(bg,axis=0)- np.size(bg[bg==0],axis=0))
b_bg_mean=np.sum(b_bg)/sayac_bg
g_bg_mean=np.sum(g_bg)/sayac_bg
r_bg_mean=np.sum(r_bg)/sayac_bg

c_b = b_bg_mean - b_fg_mean
c_g = g_bg_mean - g_fg_mean
c_r = r_bg_mean - r_fg_mean

c = ((b_bg_mean * b_fg_mean) + 
    (g_bg_mean * g_fg_mean) + 
    (r_bg_mean * r_fg_mean))/(
    (b_fg_mean * b_fg_mean) + 
    (g_fg_mean * g_fg_mean) + 
    (r_fg_mean * r_fg_mean))

for j in range(0,np.size(Negate,axis=0)):
    for i in range(0,np.size(Negate,axis=1)):
        if Negate[j,i] == 255:
            bg[j, i][0] += c_b/2 + c
            bg[j, i][1] += c_g/2 + c
            bg[j, i][2] += c_r/2 + c"""
            
stop=time.time()

time=stop-start
print time
#ekran gosterimleri
cv2.imshow('Gerçek',img_)
cv2.imshow('GaussianBlur',gauss)
cv2.imshow('Mean Shift',seg_image)
cv2.imshow('morphologyEx',morphology)
cv2.imshow('Negatif',Negate)
cv2.imshow('Edges Detection',edges)
cv2.imshow('FG',fg)
cv2.imshow('BG',bg)
cv2.imshow('BG-FG',bg_fg)

cv2.waitKey()
cv2.destroyAllWindows()