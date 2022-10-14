# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:31:50 2022

@author: gibson48
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def solve(dt, num_steps):
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    
    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        
    return xs, ys, zs

xs, ys, zs = solve(0.001, 1000000)

# Plot
#ax = plt.figure().add_subplot(projection='3d')

#ax.plot(xs, ys, zs, lw=0.5)
#ax.set_xlabel("X Axis")
#ax.set_ylabel("Y Axis")
#ax.set_zlabel("Z Axis")
#ax.set_title("Lorenz Attractor")

#axx = plt.figure().add_subplot()
#axy = plt.figure().add_subplot()
#axz = plt.figure().add_subplot()

#axx.plot(range(num_steps+1), xs, lw=0.5)
#axy.plot(range(num_steps+1), ys, lw=0.5)
#axz.plot(range(num_steps+1), zs, lw=0.5)

#plt.show()


###########################

#f, t, Zxx = signal.stft(ys, nperseg=400)
#plt.pcolormesh(t, f, np.abs(Zxx))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

data = xs
fft_size = 10
overlap_fac=0.5
fs = 10e3
 
hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
t_max = len(data) / np.float32(fs)
 
window = np.hanning(fft_size)  # our half cosine window
inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size
 
proc = np.concatenate((data, np.zeros(pad_end_size)))              # the data to process
result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result
 
for i in range(total_segments):                      # for each segment
    current_hop = hop_size * i                        # figure out the current segment offset
    segment = proc[current_hop:current_hop+fft_size]  # get the current segment
    windowed = segment * window                       # multiply by the half cosine function
    padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
    spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
    autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
    result[i, :] = autopower[:fft_size]               # append to the results array
 
#result = 20*np.log10(result)          # scale to db
result = np.clip(result, -40, 200)    # clip values

img = plt.imshow(result, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
plt.show()