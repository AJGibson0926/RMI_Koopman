# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:34:12 2022

@author: gibson48
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib import rcParams
import pdb
import csv
from scipy.integrate import odeint
from random import sample

matplotlib.rc('text',usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':18})

#######
# Load prams
#######
with open('RMI_params.pkl', 'rb') as fn:
    params = pickle.load(fn)

t = params['t']

#######
# Load solution to plot/movify
#######

##########################################################
# Plotting
##########################################################
Colorlist = ['navy','cyan','magenta','purple','gray','green','darkorange','black','pink','yellow','gray','blue']

##########################################################
# Plot 1
##########################################################
fig, ax = plt.subplots()
for idx in range(len(vplist)):   
   for alpha in range(N):
      plt.plot(sollist[idx][:,alpha],sollist[idx][:,N+alpha] ) #,color=Colorlist[alpha],lw=1.4) #, 'k', lw=3, zorder=9)linestyle='dashed'p,label="Posterior samples"

plt.xlabel("X ($mm$)", fontsize=22)
plt.ylabel("Y ($mm$)", fontsize=22)
#plt.xlim(tstart,tend)
#plt.ylim(-0.006,0.0905)
ax.yaxis.tick_left()
ax.xaxis.tick_bottom()
#plt.legend(bbox_to_anchor=(0., 1.), loc='upper left', ncol=1,fontsize=12,frameon=False)
matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22)
ratio = 1 
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
plt.tight_layout()
plt.savefig("plt_vortices.pdf",dpi=150)
plt.savefig("plt_vortices.png",dpi=150)
plt.close()

##########################################################
# Plot 2
##########################################################
fig, ax = plt.subplots()
speedlist = []
for idx in range(len(vplist)):   
   for alpha in range(N):
      speed=np.gradient(sollist[idx][:,alpha])/np.gradient(t)
      speedlist.append(speed)
      plt.plot(t,speed,color='gray',linewidth=0.4 )

jetspeed = np.zeros(len(t))
jetwidth = 0.2
for tdx in range(len(t)):   
   numInJet = 0
   for alpha in range(N):
      if np.abs(sollist[idx][tdx,alpha+N])< jetwidth:
          jetspeed[tdx] = jetspeed[tdx]+ speedlist[alpha][tdx] 
          numInJet = numInJet+1
   jetspeed[tdx] = jetspeed[tdx]/numInJet

#pdb.set_trace()
plt.plot(t,jetspeed,color='black',label='Jet Speed' )
plt.plot(t,speed,color='gray',linewidth=0.4,label='Individual Vortex'  )

plt.xlabel("t ($\\mu s$)", fontsize=22)
plt.ylabel("$v_x$ ($mm/\\mu s$)", fontsize=22)
plt.xlim(0.0,t[-1])
#plt.ylim(-0.006,0.0905)
ax.yaxis.tick_left()
ax.xaxis.tick_bottom()
plt.legend(bbox_to_anchor=(0., 1.), loc='upper left', ncol=1,fontsize=12,frameon=False)
matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22)
ratio = 1 
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
plt.tight_layout()
plt.savefig("plt_velocities_per_vortex.pdf",dpi=150)
plt.savefig("plt_velocities_per_vortex.png",dpi=150)
#plt.show()
plt.close()

##########################################################
# Movie
##########################################################
if True:
   numFrames = round(len(t)/5)
   for jdx in range(numFrames+1):   
      fig, ax = plt.subplots()
      for idx in range(len(vplist)):   
         endIdx = int( (jdx)/float(numFrames) * len(sollist[idx][:,alpha]) )
         print(endIdx)
         for alpha in range(N):
            plt.plot(sollist[idx][:endIdx,alpha],sollist[idx][:endIdx,N+alpha],color='gray',linestyle='dashed' ) 
            if sollist[idx][0,N+alpha]>0:
               if jdx<numFrames:
                  plt.plot(sollist[idx][endIdx,alpha],sollist[idx][endIdx,N+alpha],color='blue',marker='o',linestyle='None' )
               else:
                  plt.plot(sollist[idx][endIdx-1,alpha],sollist[idx][endIdx-1,N+alpha],color='blue',marker='o',linestyle='None' )
            else:
               if jdx<numFrames:
                  plt.plot(sollist[idx][endIdx,alpha],sollist[idx][endIdx,N+alpha],color='red',marker='o',linestyle='None' )
               else:
                  plt.plot(sollist[idx][endIdx-1,alpha],sollist[idx][endIdx-1,N+alpha],color='red',marker='o',linestyle='None' )
                  
      plt.xlabel("X ($mm$)", fontsize=22)
      plt.ylabel("Y ($mm$)", fontsize=22)
      plt.xlim(0.,2.4)
      plt.ylim(-1.05,1.05)
      ax.yaxis.tick_left()
      ax.xaxis.tick_bottom()
      if jdx<numFrames:
         plt.title('Time = {:.2f} $\\mu $s'.format(t[endIdx]))
      else:
         plt.title('Time = {:.2f} $\\mu $s'.format(t[endIdx-1]))
         
      #plt.title('Time = '+str(t[endIdx]))
      #plt.legend(bbox_to_anchor=(0., 1.), loc='upper left', ncol=1,fontsize=12,frameon=False)
      matplotlib.rc('xtick', labelsize=22) 
      matplotlib.rc('ytick', labelsize=22)
      ratio = 1 
      xleft, xright = ax.get_xlim()
      ybottom, ytop = ax.get_ylim()
      ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
      plt.tight_layout()
      plt.savefig("images/jet{:04}.png".format(jdx),dpi=150)
      plt.close()

##########################################################
# Plot 2
##########################################################
fig, ax = plt.subplots()
for idx in range(len(vplist)):   
   #plt.plot(ylist,vplist[idx],color=Colorlist[idx],lw=1.4) #, 'k', lw=3, zorder=9)linestyle='dashed'p,label="Posterior samples"
   plt.plot(ylist,vplist[idx],lw=1.4) #, 'k', lw=3, zorder=9)linestyle='dashed'p,label="Posterior samples" *(1.25-ylist)
plt.plot(ylist,vplist[0],lw=2.4,color='black') #, 'k', lw=3, zorder=9)linestyle='dashed'p,label="Posterior samples" *(1.25-ylist)
plt.xlabel("Y @ X=0($mm$)", fontsize=18)
plt.ylabel("v ($mm/\\mu s$)", fontsize=18)
#plt.xlim(tstart,tend)
#plt.ylim(-0.006,0.0905)
ax.yaxis.tick_left()
ax.xaxis.tick_bottom()
plt.legend(bbox_to_anchor=(0., 1.), loc='upper left', ncol=1,fontsize=12,frameon=False)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
ratio = 1 
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
plt.tight_layout()
plt.savefig("plt_profile.pdf",dpi=150)
plt.savefig("plt_profile.png",dpi=150)
plt.close()
#plt.show()

##########################################################
# Plot 3
##########################################################
if False:
   mlist = np.array(mlist)
   paramlist =np.transpose(np.array(paramlist))
   labels = ["Vortex Strength (mm$^2$/us)","Vortex Period (mm)","Vortex Standoff (mm)","Offset (mm)"]
   for idx in range(len(paramlist)):
      const = mlist[0]*np.ones(len(mlist))
      fig, ax = plt.subplots()
      plt.scatter(paramlist[idx],mlist,lw=1.4) #, 'k', lw=3, zorder=9)linestyle='dashed'p,label="Posterior samples" *(1.25-ylist)
      plt.plot(paramlist[idx],const,lw=1.4,color='black') #, 'k', lw=3, zorder=9)linestyle='dashed'p,label="Posterior samples" *(1.25-ylist)
      plt.xlabel(labels[idx], fontsize=18)
      plt.ylabel("$\int|v|^2 dx $", fontsize=18)
      #plt.xlim(tstart,tend)
      #plt.ylim(-0.006,0.0905)
      ax.yaxis.tick_left()
      ax.xaxis.tick_bottom()
      plt.legend(bbox_to_anchor=(0., 1.), loc='upper left', ncol=1,fontsize=12,frameon=False)
      matplotlib.rc('xtick', labelsize=18) 
      matplotlib.rc('ytick', labelsize=18)
      ratio = 1 
      xleft, xright = ax.get_xlim()
      ybottom, ytop = ax.get_ylim()
      ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
      plt.tight_layout()
      plt.savefig("plt_metric_"+str(idx)+".pdf",dpi=150)
      plt.savefig("plt_metric_"+str(idx)+".png",dpi=150)
#plt.show()