#!/usr/bin/env python
# coding: utf-8
"""
This script segments a given track using the Constrained Clustering method
described here:

Levy, M., & Sandler, M. (2008). Structural Segmentation of Musical Audio by
Constrained Clustering. IEEE Transactions on Audio, Speech, and Language
Processing, 16(2), 318–326. doi:10.1109/TASL.2007.910781
"""

import logging
import numpy as np
import os

if PLOT:
	'''
	If visualisation is unavailable (running on the server etc.), import the following:
	import matplotlib
	matplotlib.use('Agg')
	'''
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	SAVETO = '/Users/mitian/Documents/hg/phd-docs/thesis/notebooks/fig'

# Local stuff
from msaf.algorithms.interface import SegmenterInterface
try:
	from msaf.algorithms.cc import cc_segmenter
except:
	pass


def plot_seg(hmmStates, est_idxs, est_labels, annot_idxs=None, annot_labels=None, show=False):
	"""Ploting function."""

	length = len(hmmStates)	
	
	if annot_idxs==None:
		gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
		plt.figure(figsize=(7,5))
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1], sharex=ax0)
		ax2 = plt.subplot(gs[2], sharex=ax0)
		
		ax0.plot(np.arange(length), hmmStates, 'go-')
		ax0.set_title('HMM states', fontsize=22)
		ax0.set_ylabel('States', fontsize=20)
		
		ax1.plot(np.arange(length), est_labels, 'r<-')
		ax1.vlines(est_idxs, 0-0.5, rank-0.5, 'k')
		ax1.set_title('Constrianed clustering states', fontsize=22)
		ax1.set_ylabel('States', fontsize=20)
		
		if not annot_labels == None:
			ax2.plot(np.arange(length), annot_labels, 'r<-')
		ax2.vlines(annot_idxs, 0-0.5, rank-0.5, 'k')
		ax2.set_title('Annotations', fontsize=22)
		ax2.set_ylabel('States', fontsize=20)	
		ax2.xaxis.set_ticks_position('bottom') 
		
		plt.xlabel('Time frame', fontsize=20)
		
	else:
		gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
		plt.figure(figsize=(7,5))
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1], sharex=ax0)
		
		ax0.plot(np.arange(length), hmmStates, 'go-')
		ax0.set_title('HMM states', fontsize=22)
		ax0.set_ylabel('States', fontsize=20)
		
		ax1.plot(np.arange(length), est_labels, 'r<-')
		ax1.vlines(est_idxs, 0-0.5, rank-0.5, 'k')
		ax1.set_title('Constrianed clustering states', fontsize=22)
		ax1.set_ylabel('States', fontsize=20)
		
		plt.xlabel('Time frame', fontsize=20)
	
	if show:
		plt.show()
	else:
		plt.savefig(join(SAVETO+'CC.pdf'), format='pdf')
	
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
		min_frames = 15
		# Preprocess to obtain features, times, and input boundary indeces
		F = self._preprocess(normalize=True)


		# Check if the cc module is compiled
		try:
			cc_segmenter
		except:
			logging.warning("CC not compiled, returning empty segmentation")
			if self.in_bound_idxs is None:
				return np.array([0, F.shape[0] - 1]), [0]
			else:
				return self.in_bound_idxs, [0] * (len(self.in_bound_idxs) - 1)

		if F.shape[0] > min_frames:
			if self.feature_str == "hpcp" or self.feature_str == "tonnetz" or \
					self.feature_str == "cqt":
				is_harmonic = True
			elif self.feature_str == "mfcc" or self.feature_str == 'gmt':
				is_harmonic = False
			else:
				raise RuntimeError("Feature type %s is not valid" %
								   self.feature_str)

			in_bound_idxs = self.in_bound_idxs
			if self.in_bound_idxs is None:
				in_bound_idxs = []
			
			# If plot==True, return hmmStates as an intermediate step result (return num == 3)
			# Otherwise return two values est_idxs, est_labels as a tuple
			if F.shape[0] > 2 and \
					(len(in_bound_idxs) > 2 or len(in_bound_idxs) == 0):
				cc_res = cc_segmenter.segment(
					is_harmonic, self.config["nHMMStates"],
					self.config["nclusters"], self.config["neighbourhoodLimit"],
					self.anal["sample_rate"], F, in_bound_idxs, plot=PLOT)
			else:
				est_idxs = in_bound_idxs
				est_labels = [0]
				hmmStates = np.zeros(F.shape[0] - 1)
			
			if PLOT:
				(est_idxs, est_labels) = cc_res[:2]
			else:
				(est_idxs, est_labels, hmmStates) = cc_res
		else:
			# The track is too short. We will only output the first and last
			# time stamps
			est_idxs = np.array([0, F.shape[0] - 1])
			est_labels = [1]
			hmmStates = np.zeros(F.shape[0] - 1)
		
		# Make sure that the first and last boundaries are included
		assert est_idxs[0] == 0	 and est_idxs[-1] == F.shape[0] - 1

		# Post process estimations
		est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

		# import pylab as plt
		# plt.imshow(F.T, interpolation="nearest", aspect="auto")
		# plt.show()
		# plt.savefig(os.path.join(plot_save, '.pdf'), format='pdf')
		
		if PLOT:
			plot_seg(hmmStates, est_idxs, est_labels)	
			
		return est_idxs, est_labels
