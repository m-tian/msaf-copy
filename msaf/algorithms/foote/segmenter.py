#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using the Foote
method:

Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio
Novelty. In Proc. of the IEEE International Conference of Multimedia and Expo
(pp. 452–455). New York City, NY, USA.
"""
import librosa
import logging
import sys, os
from os.path import join, basename, splitext
import numpy as np
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters
import pylab as plt

import msaf
from msaf.algorithms.interface import SegmenterInterface


def median_filter(X, M=8):
	"""Median filter along the first axis of the feature matrix X."""
	for i in xrange(X.shape[1]):
		X[:, i] = filters.median_filter(X[:, i], size=M)
	return X


def compute_gaussian_krnl(M):
	"""Creates a gaussian kernel following Foote's paper."""
	g = signal.gaussian(M, M / 3., sym=True)
	G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
	G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
	G[:M / 2, M / 2:] = -G[:M / 2, M / 2:]
	return G


def compute_ssm(X, metric="seuclidean"):
	"""Computes the self-similarity matrix of X."""
	D = distance.pdist(X, metric=metric)
	D = distance.squareform(D)
	D /= D.max()
	return 1 - D


def compute_nc(X, G):
	"""Computes the novelty curve from the self-similarity matrix X and
		the gaussian kernel G."""
	N = X.shape[0]
	M = G.shape[0]
	nc = np.zeros(N)

	for i in xrange(M / 2, N - M / 2 + 1):
		nc[i] = np.sum(X[i - M / 2:i + M / 2, i - M / 2:i + M / 2] * G)

	# Normalize
	nc += nc.min()
	nc /= nc.max()
	return nc


def pick_peaks(nc, L=16):
	"""Obtain peaks from a novelty curve using an adaptive threshold."""
	offset = nc.mean() / 20.

	nc = filters.gaussian_filter1d(nc, sigma=4)	 # Smooth out nc

	th = filters.median_filter(nc, size=L) + offset
	#th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset

	peaks = []
	for i in xrange(1, nc.shape[0] - 1):
		# is it a peak?
		if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
			# is it above the threshold?
			if nc[i] > th[i]:
				peaks.append(i)
	#plt.plot(nc)
	#plt.plot(th)
	#for peak in peaks:
		#plt.axvline(peak)
	#plt.show()

	return peaks


def plot_seg(nc, detection=None, annotation=None, show=False):
	
	length = len(nc)
	if detection==None:
		# Empty detection. Plot the first and last instance only.
		detection = [0, length-1]
	
	if annotation == None:
		# Plot single pane
		plt.figure(figsize=(7,2))
		plt.plot(np.linspace(0, length-1, length), nc)
		plt.vlines(detection, 0, 1, 'k')
		
	else:
		# Plot duo panes
		gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
		plt.figure(figsize=(7,5))
	
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1], sharex=ax0)
	
		ax0.plot(np.linspace(0, length-1, length), nc)
		ax0.vlines(detection, 0, 1, 'k')
	   
		ax0.set_xlim([0, length])
		ax0.set_ylim([0, 1])
		ax0.get_xaxis().set_visible(False)
		ax0.set_title('Nolvety curve', fontsize=22)
		ax0.set_ylabel('Amplitude', fontsize=20)
	
		ax1.set_xlim([0, length])
		ax1.set_ylim([0, 1])
		ax1.vlines(annotation, 0, 1, 'r')
	
		ax1.get_xaxis().set_visible(False)
		ax1.set_title('Annotations', fontsize=22)

		plt.xlabel('Time frame', fontsize=20)
	
	if show:
		plt.show()
	else:
		plt.savefig(join(SAVETO+'FOOTE.pdf'), format='pdf')
	
	plt.close()


class Segmenter(SegmenterInterface):
	def processFlat(self):
		"""Main process.
		Returns
		-------
		est_idxs : np.array(N)
			Estimated indeces the segment boundaries in frames.
		est_labels : np.array(N-1)
			Estimated labels for the segments.
		"""
		# Preprocess to obtain features
		F = self._preprocess()

		# Make sure that the M_gaussian is even
		if self.config["M_gaussian"] % 2 == 1:
			self.config["M_gaussian"] += 1

		# Median filter
		F = median_filter(F, M=self.config["m_median"])
		#plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

		# Self similarity matrix
		S = compute_ssm(F)

		# Compute gaussian kernel
		G = compute_gaussian_krnl(self.config["M_gaussian"])
		#plt.imshow(S, interpolation="nearest", aspect="auto"); plt.show()

		# Compute the novelty curve
		nc = compute_nc(S, G)

		# Find peaks in the novelty curve
		est_idxs = pick_peaks(nc, L=self.config["L_peaks"])

		# Add first and last frames
		est_idxs = np.concatenate(([0], est_idxs, [F.shape[0] - 1]))

		# Empty labels
		est_labels = np.ones(len(est_idxs) - 1) * -1

		# Post process estimations
		est_idxs, est_labels = self._postprocess(est_idxs, est_labels)


		return est_idxs, est_labels
		# plt.figure(1)
		# plt.plot(nc);
		# [plt.axvline(p, color="m") for p in est_bounds]
		# [plt.axvline(b, color="g") for b in ann_bounds]
		# plt.figure(2)
		# plt.imshow(S, interpolation="nearest", aspect="auto")
		# [plt.axvline(b, color="g") for b in ann_bounds]
		# plt.show()
