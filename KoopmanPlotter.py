# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 18:27:35 2022

@author: gibson48
"""
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

##### GRAPHICS #####
class KoopmanPlotter:
    
    # provided dataset structure: [N,M]x[T,K,P]
    # N neural nets and M datasets (must be TVTs: Training-Validation-Test type)
    # T time-steps, K is the number of trajectories, and P the number of coordinates/components of motion.

    def reweave(x):
        x_new = np.hstack([x[:,0::2], x[:,1::2]])
    
        return x_new


    def grab_fig(NTs, NNs, option):
        if option=='grid':
            xs = max(6*NTs, 15)
            ys = max(6*NNs - 1, 15)
            fig, ax = plt.subplots(NNs, NTs, figsize=(xs, ys), dpi=150)
        elif option=='overlay':
            fig, ax = plt.subplots(1, figsize=(15, 15), dpi=150)
            
        return fig, ax
    
    
    def grab_axl(ax,col):
        if type(ax)==np.ndarray:
            if len(ax.shape)>1:
                axl = ax[col,:].flat
            else:
                axl = [ax[col]]
        elif type(ax)==list:
            axl = ax
        else:
            axl = [ax]
        
        return axl
    
    
    def animate_traj(self, traj, t, option='grid', scale='off'):
        # Generate movie frames
        T = traj.shape[0]
        numFrames = round(T/5)
        for tdx in range(numFrames+1):
            endIdx = int((tdx)/float(numFrames) * T)
    
            framedict = {'frm': tdx, 'endIdx': endIdx, 'numFrames': numFrames, 'tSpan': t}
            self.plot_single_frame(traj, framedict=framedict, scale=scale)
    
            plt.tight_layout()
            print('Saving frame %s' % tdx)
            plt.savefig("images/jetsframe{:04}.png".format(tdx))
            plt.close()
        
        def animate(i):
            plt.clf()
            im = plt.imread("images/jetsframe{:04}.png".format(i))
            plt.axis('off')
            plt.imshow(im)
        
        fig, ax = plt.subplots(1,figsize=(12, 8))
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
        
        print('Saving movie')
        anim = FuncAnimation(plt.gcf(), animate, frames=numFrames, interval=(2000.0/numFrames))
        anim.save('images/gifs/jets.gif', writer='pillow')
        print('Done')


    def plot_single_frame(self, traj, framedict, option='grid', scale='off'):
        # Cycle through the trajectories in the dataset provided and plot one frame for all of it.
        N = int(np.floor(traj.shape[1]/3)) # Number of vortices
        
        frm = framedict['frm']
        numFrames = framedict['numFrames']
        endIdx = framedict['endIdx']
        tSpan = framedict['tSpan']

        if frm<numFrames:
            tidx = endIdx
        else:
            tidx = endIdx-1

        # Initialise the subplot function using number of rows and columns
        fig, ax = plt.subplots(1, figsize=(15, 15), dpi=150)
        fig.suptitle('Time = {:.2f} $\\mu $s'.format(tSpan[tidx]))

        x = traj[:,0:N]
        y = traj[:,N:2*N]
        G = traj[:,2*N:]
        for alpha in range(N):
            ## COLOR SCHEME
            if option=='grid':
                if G[tidx, alpha]>0:
                    clr = 'blue'
                else:
                    clr = 'red'

            elif option=='overlay':
                clrs = ['gray', 'navy', 'purple']
                clr = clrs[0]
        
            # if overlay style, do not show history
            if option=='grid':
                ax.plot(x[:endIdx,alpha], y[:endIdx,alpha], color='gray', linestyle='dashed')
    
            ax.plot(x[tidx,alpha], y[tidx,alpha], color=clr, marker='o', linestyle='None')
    
            ratio = 1
            ax.set(xlabel='X ($mm$)', ylabel='Y ($mm$)')
            if scale=='on':
                ax.set_xlim(-2,4)
                ax.set_ylim(-2,2)
    
            ax.label_outer()
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()
            ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    
        matplotlib.rc('xtick', labelsize=22) 
        matplotlib.rc('ytick', labelsize=22)
        
        return fig, ax
    
    def plt_rainbow_trajs(self, trajlist, dataset, option='dyn_states', option2='grid'):
        # Must be a single dataset, but may be multiple neural networks
        # Cycle through the trajectories and the dataset provided and build rainbow plots for all of it.
        data = dataset[0]
        C = data.shape[2]
        NTs = len(trajlist)
        NNs = len(dataset)
        N = int(np.floor(C/2)) # number of vortices
    
        # Initialise the subplot function using number of rows and columns
        fig, ax = self.grab_fig(NTs, NNs, option2)
    
        for tr, traj in enumerate(trajlist):
            axl = self.grab_axl(ax,tr)
            for idx, axs in enumerate(axl):
                if option=='dyn_states':
                    for k in range(N):
                        axs.scatter(data[:,traj,k],data[:,traj,N+k])
                elif option=='koop_states':
                    for k in range(C):
                        axs.plot(data[:,traj,k])


    def plot_Koopman_states(self, model='RMI', animate='off', style='grey'):
        if animate=='on':
            return

        else:
            if model=='RMI':
                return

        return


    def plot_dynamic_states(self, model='RMI', animate='off', style='grey'):
        if animate=='on':
            return

        else:
            if model=='RMI':
                return

        return


    def plot_time_series(self, x, animate='off', style='rainbow'):
        if animate=='on':
            return

        else:
            plt.plot(x)


    def histograms(self, avgs, maxs):
        fig, ax = plt.subplots(2,2,figsize=(15,14))

        a=val_max_imp
        b=pred_max_imp
        bins=np.histogram(np.hstack((a,b)), bins='auto')[1] #get the bin edges
        ax[0,0].hist(val_max_imp, bins=bins)
        ax[0,0].hist(pred_max_imp, bins=bins, alpha=0.5)
        
        a=val_avg_imp
        b=pred_avg_imp
        bins=np.histogram(np.hstack((a,b)), bins='auto')[1] #get the bin edges
        ax[0,1].hist(val_avg_imp, bins=bins)
        ax[0,1].hist(pred_avg_imp, bins=bins, alpha=0.5)
        
        a=val_max_ang_imp
        b=pred_max_ang_imp
        bins=np.histogram(np.hstack((a,b)), bins='auto')[1] #get the bin edges
        ax[1,0].hist(val_max_ang_imp, bins=bins)
        ax[1,0].hist(pred_max_ang_imp, bins=bins, alpha=0.5)
        
        a=val_avg_ang_imp
        b=pred_avg_ang_imp
        bins=np.histogram(np.hstack((a,b)), bins='auto')[1] #get the bin edges
        ax[1,1].hist(val_avg_ang_imp, bins=bins)
        ax[1,1].hist(pred_avg_ang_imp, bins=bins, alpha=0.5)
        
        for axs in ax.flatten():
            axs.set_yscale('log')
            axs.legend(["Ground Truth", "Prediction"])
            axs.set_ylabel('Number of ICs')
        
        fig.suptitle('Histograms of Jet Strengths')
        ax[0,0].set_xlabel(r'Max Linear Impulse $\Gamma_{\alpha} x_{\alpha}$')
        ax[0,1].set_xlabel(r'Avg Linear Impulse $\Gamma_{\alpha} x_{\alpha}$')
        ax[1,0].set_xlabel(r'Max Angular Impulse $\Gamma_{\alpha} (x_{\alpha}^2 + y_{\alpha}^2)$')
        ax[1,1].set_xlabel(r'Avg Angular Impulse $\Gamma_{\alpha} (x_{\alpha}^2 + y_{\alpha}^2)$')
        
        plt.tight_layout()


    def companion_plots(self, ks, dyn_sts, animate='off', frames=dict(), scale='off'):
        coords = int(dyn_sts.shape[1]/3)
        dyn_x = dyn_sts[:,0:coords]
        dyn_y = dyn_sts[:,coords:2*coords]
    
        coords = int(ks.shape[1]/2)
        rks_x = ks[:,0:coords]
        rks_y = ks[:,coords:2*coords]
    
        if animate=='on':
            fsz = (16,8)
            frm = frames['frm']
            numFrames = frames['numFrames']
            endIdx = frames['endIdx']
            tSpan = frames['tSpan']
    
            if frm<numFrames:
                nidx = endIdx
            else:
                nidx = endIdx-1
        else:
            fsz = (20,8)
    
        if isinstance(ks, list):
            rows = len(ks)
            fig, ax = plt.subplots(rows, 2, figsize=fsz, dpi=150)
        else:
            rows = 1
            fig, ax = plt.subplots(rows, 2, figsize=fsz, dpi=150)
            ax = ax[:,np.newaxis]
    
        fs = 48
        for row in np.arange(rows):
            if animate=='off':
                sm = ax[0,row].scatter(rks_x, rks_y, c=range(rks_x.shape[0]*rks_x.shape[1]), cmap='viridis')
                ax[1,row].scatter(dyn_x, dyn_y, c=range(dyn_x.shape[0]*dyn_x.shape[1]), cmap='viridis')
    
                cbar = fig.colorbar(sm, ax=ax[1,row])
                cbar.set_label('timestep $t_k$', fontsize=fs)
                cbar = fig.colorbar(sm, ax=ax[0,row])
                cbar.set_label('timestep $t_k$', fontsize=fs)
    
            elif animate=='on':
                sm = ax[0,row].plot(rks_x[:endIdx], rks_y[:endIdx], color='gray', linestyle='dashed')
                ax[0,row].plot(rks_x[nidx], rks_y[nidx], color='navy', marker='o', linestyle='None')
                
                ax[1,row].plot(dyn_x[:endIdx], dyn_y[:endIdx], color='gray', linestyle='dashed')
                ax[1,row].plot(dyn_x[nidx], dyn_y[nidx], color='navy', marker='o', linestyle='None')
    
            ax[0,row].set_title('Koopman states', fontsize=fs)
            ax[0,row].set_xlabel('real($\psi_n$)', fontsize=fs)
            ax[0,row].set_ylabel('imag($\psi_n$)', fontsize=fs)
    
            ax[1,row].set_title('dynamical states', fontsize=fs)
            ax[1,row].set_xlabel('x', fontsize=fs)
            ax[1,row].set_ylabel('y', fontsize=fs)
    
            if scale=='on':
                ax[0,row].set_xlim(-0.01,0.07)
                ax[0,row].set_ylim(-0.01,0.09)
                ax[1,row].set_xlim(-0.1,3.5)
                ax[1,row].set_ylim(-1.5,1.5)
    
        matplotlib.rc('text',usetex=True)
        matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':18})
        plt.tight_layout()