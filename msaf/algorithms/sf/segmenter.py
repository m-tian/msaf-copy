#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using the Serrà
method:

Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2012). Unsupervised
Detection of Music Boundaries by Time Series Structure Features.
In Proc. of the 26th AAAI Conference on Artificial Intelligence
(pp. 1613–1619).

Toronto, Canada.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import logging
import numpy as np
import sys, os
from os.path import join, basename, splitext
import librosa
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters

from msaf.algorithms.interface import SegmenterInterface

PLOT = True

if PLOT:
	'''
	If visualisation is unavailable (running on the server etc.), import the following:
	import matplotlib
	matplotlib.use('Agg')
	'''
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	SAVETO = '/Users/mitian/Documents/hg/phd-docs/thesis/notebooks/fig'

def median_filter(X, M=8):
	"""Median filter along the first axis of the feature matrix X."""
	for i in xrange(X.shape[1]):
		X[:, i] = filters.median_filter(X[:, i], size=M)
	return X


def gaussian_filter(X, M=8, axis=0):
	"""Gaussian filter along the first axis of the feature matrix X."""
	for i in xrange(X.shape[axis]):
		if axis == 1:
			X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
		elif axis == 0:
			X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)
	return X


def compute_gaussian_krnl(M):
	"""Creates a gaussian kernel following Serra's paper."""
	g = signal.gaussian(M, M / 3., sym=True)
	G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
	G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
	G[:M / 2, M / 1:] = -G[:M / 2, M / 1:]
	return G


def compute_ssm(X, metric="seuclidean"):
	"""Computes the self-similarity matrix of X."""
	D = distance.pdist(X, metric=metric)
	D = distance.squareform(D)
	D /= D.max()
	return 1 - D


def compute_nc(X):
	"""Computes the novelty curve from the structural features."""
	N = X.shape[0]
	# nc = np.sum(np.diff(X, axis=0), axis=1) # Difference between SF's

	nc = np.zeros(N)
	for i in xrange(N - 1):
		nc[i] = distance.euclidean(X[i, :], X[i + 1, :])

	# Normalize
	nc += np.abs(nc.min())
	nc /= nc.max()
	return nc


def pick_peaks(nc, L=16, offset_denom=0.1):
	"""Obtain peaks from a novelty curve using an adaptive threshold."""
	offset = nc.mean() * float(offset_denom)
	th = filters.median_filter(nc, size=L) + offset
	#th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset
	#import pylab as plt
	#plt.plot(nc)
	#plt.plot(th)
	#plt.show()
	# th = np.ones(nc.shape[0]) * nc.mean() - 0.08
	peaks = []
	for i in xrange(1, nc.shape[0] - 1):
		# is it a peak?
		if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
			# is it above the threshold?
			if nc[i] > th[i]:
				peaks.append(i)
	return peaks


def circular_shift(X):
	"""Shifts circularly the X squre matrix in order to get a
		time-lag matrix."""
	N = X.shape[0]
	L = np.zeros(X.shape)
	for i in xrange(N):
		L[i, :] = np.asarray([X[(i + j) % N, j] for j in xrange(N)])
	return L


def embedded_space(X, m, tau=1):
	"""Time-delay embedding with m dimensions and tau delays."""
	N = X.shape[0] - int(np.ceil(m))
	Y = np.zeros((N, int(np.ceil(X.shape[1] * m))))
	for i in xrange(N):
		# print X[i:i+m,:].flatten().shape, w, X.shape
		# print Y[i,:].shape
		rem = int((m % 1) * X.shape[1])	 # Reminder for float m
		Y[i, :] = np.concatenate((X[i:i + int(m), :].flatten(),
								 X[i + int(m), :rem]))
	return Y


def plot_seg(X, R, nc, detection=None, annotation=None, show=False):
	
	length = len(nc)

	gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
	plt.figure(figsize=(7,10))
	
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1], sharex=ax0)
	# ax2 = plt.subplot(gs[2], sharex=ax0)
	
	# X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2.0 - 1 # Normalise to [-1,1]
	# ax0.plot(np.linspace(0, length-1, length), X[:, ::6]) #Plot every 4th series for visualisation purpose
	# # ax0.imshow(X)
	# ax0.set_xlim([0, length])
	# ax0.set_xlim([-1, 1])
	# ax0.get_xaxis().set_visible(False)
	# ax0.set_title('Multi-dimensional time series', fontsize=22)
	# ax0.set_ylabel('Amplitude', fontsize=20)
	
	ax0.imshow(R, cmap='Greys')
	ax0.set_xlim([0, length])
	ax0.set_ylim([0, length])
	ax0.get_xaxis().set_visible(False)
	ax0.set_title('Recurrence plot', fontsize=22)

	ax1.plot(np.linspace(0, length-1, length), nc)
	if detection==None:
		# Empty detection. Plot the first and last instance only.
		detection = [0, length-1]
	ax1.vlines(detection, 0, 1, 'k')
	if not annotation==None:
		ax1.vlines(annotation, 0, 1, 'r')

	ax1.set_title('Novelty curve', fontsize=22)
	ax1.set_ylabel('Novelty score', fontsize=20)
	ax1.xaxis.set_ticks_position('bottom') 
	ax1.get_yaxis().set_visible(False)
	plt.xlabel('Time frame', fontsize=20)
	
	if show:
		plt.show()
	else:
		plt.savefig(join(SAVETO+'SF.pdf'), format='pdf')
	
	plt.close()


class Segmenter(SegmenterInterface):
	def processFlat(self):
		"""Main process.
		Returns
		-------
		est_idxs : np.array(N)
			Estimated times for the segment boundaries in frame indeces.
		est_labels : np.array(N-1)
			Estimated labels for the segments.
		"""
		# Structural Features params
		Mp = self.config["Mp_adaptive"]	  # Size of the adaptive threshold for
										  # peak picking
		od = self.config["offset_thres"]  # Offset coefficient for adaptive
										  # thresholding

		M = self.config["M_gaussian"]	  # Size of gaussian kernel in beats
		m = self.config["m_embedded"]	  # Number of embedded dimensions
		k = self.config["k_nearest"]	  # k*N-nearest neighbors for the
										  # recurrence plot

		# Preprocess to obtain features, times, and input boundary indeces
		F = self._preprocess()

		# Check size in case the track is too short
		if F.shape[0] > 20:

			if self.framesync:
				red = 0.1
				F_copy = np.copy(F)
				F = librosa.feature.sync(F.T, np.linspace(0, F.shape[0], num=F.shape[0] * red), pad=False).T

			# Emedding the feature space (i.e. shingle)
			E = embedded_space(F, m)
			# plt.imshow(E.T, interpolation="nearest", aspect="auto"); plt.show()

			# Recurrence matrix
			R = librosa.segment.recurrence_matrix(
				E.T,
				k=k * int(F.shape[0]),
				width=1,  # zeros from the diagonal
				metric="seuclidean",
				sym=True).astype(np.float32)

			# Circular shift
			L = circular_shift(R)
			#plt.imshow(L, interpolation="nearest", cmap=plt.get_cmap("binary"))
			#plt.show()

			# Obtain structural features by filtering the lag matrix
			SF = gaussian_filter(L.T, M=M, axis=1)
			SF = gaussian_filter(L.T, M=1, axis=0)
			# plt.imshow(SF.T, interpolation="nearest", aspect="auto")
			#plt.show()

			# Compute the novelty curve
			nc = compute_nc(SF)

			# Find peaks in the novelty curve
			est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)

			# Re-align embedded space
			est_bounds = np.asarray(est_bounds) + int(np.ceil(m / 2.))

			if self.framesync:
				est_bounds /= red
				F = F_copy
		else:
			est_bounds = []

		# Add first and last frames
		est_idxs = np.concatenate(([0], est_bounds, [F.shape[0] - 1]))
		est_idxs = np.unique(est_idxs)

		assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1

		# Empty labels
		est_labels = np.ones(len(est_idxs) - 1) * - 1

		# Post process estimations
		est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

		# plt.figure(1)
		# plt.plot(nc);
		# [plt.axvline(p, color="m", ymin=.6) for p in est_bounds]
		# [plt.axvline(b, color="b", ymax=.6, ymin=.3) for b in brian_bounds]
		# [plt.axvline(b, color="g", ymax=.3) for b in ann_bounds]
		# plt.show()

		if PLOT:
			plot_seg(E, R, nc, detection=est_idxs)
			
		return est_idxs, est_labels
