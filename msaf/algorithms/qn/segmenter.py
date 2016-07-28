#!/usr/bin/env python
# encoding: utf-8
"""
segmenter.py

Created by mi tian on 2015-05-22.
Copyright (c) 2015 Queen Mary University of London. All rights reserved.
"""

import sys
import os
from os.path import join, basename, splitext
import numpy as np
from scipy.signal import correlate2d, convolve2d
from scipy.spatial import distance

from msaf.algorithms.interface import SegmenterInterface

from PeakPickerUtil import PeakPicker


def getNoveltyCurve(ssm, kernel_size, normalise=False):
	'''Return novelty score from ssm.'''

	kernel_size = int(np.floor(kernel_size/2.0) + 1)
	stripe = getDiagonalSlice(ssm, kernel_size)
	kernel = gaussian_kernel(kernel_size)
	xc = convolve2d(stripe,kernel,mode='same')
	xc[abs(xc)>1e+10]=0.00001			
	
	novelty = xc[int(np.floor(xc.shape[0]/2.0)),:]	
	novelty = [0.0 if (np.isnan(x) or np.isinf(x) or x > 1e+100) else x for x in novelty]
	
	if normalise:
		novelty = (novelty - np.min(novelty)) / (np.max(novelty) - np.min(novelty))
	return novelty	

def getNoveltyDiff(novelty, N=1, relative=False):
	'''Return the second order differece in the novelty curve.'''
	
	diff = np.zeros_like(novelty)
	diff[:-N] = np.diff(novelty, n=N)
	
	if relative:
		diff /= novelty 
		
	return diff
	
def getDiagonalSlice(ssm, width):
	''' Return a diagonal stripe of the ssm given its width, with 45 degrees rotation. 
	Note: requres 45 degrees rotated kernel also.'''
	w = int(np.floor(width/2.0))
	length = len(np.diagonal(ssm))
	stripe = np.zeros((2*w+1,length))
	# print 'diagonal', length, w, stripe.shape
	for i in xrange(-w, w+1) :		
		stripe[w+i,:] = np.hstack(( np.zeros(int(np.floor(abs(i)/2.0))), np.diagonal(ssm,i), np.zeros(int(np.ceil(abs(i)/2.0))) ))
	return stripe

def gaussian_kernel(size):
	'''Create a gaussian tapered 45 degrees rotated checkerboard kernel.'''
	n = float(np.ceil(size / 2.0))
	kernel = np.zeros((size,size))
	for i in xrange(1,size+1) :
		for j in xrange(1,size+1) :
			gauss = np.exp( -4.0 * (np.square( (i-n)/n ) + np.square( (j-n)/n )) )
			# gauss = 1			
			if np.logical_xor( j - n > np.floor((i-n) / 2.0), j - n > np.floor((n-i) / 2.0) ) :
				kernel[i-1,j-1] = -gauss
			else:
				kernel[i-1,j-1] = gauss
				
	return kernel	

def compute_ssm(X, metric="seuclidean"):
	"""Computes the self-similarity matrix of X."""
	D = distance.pdist(X, metric=metric)
	D = distance.squareform(D)
	D /= D.max()
	return 1 - D	


class Segmenter(SegmenterInterface):
	
	'''The peak picker util to detect boundaries from the novelty curve.'''

	def processFlat(self):
		'''
		Returns:
			novelty (np 1d array): Raw novelty curve computed from the SSM using Foote method
			smoothed novelty curve (np 1d array): post-processed novelty curve
			novelty_peaks (list): detected boundaries
		'''
		
		peak_picker = PeakPicker()
		
		
		peak_picker.params.alpha = self.config["alpha"]
		peak_picker.params.delta = self.config["delta"]
		peak_picker.params.rawSensitivity = self.config["rawSensitivity"]
		peak_picker.params.preWin = self.config["preWin"]
		peak_picker.params.postWin = self.config["postWin"]
		peak_picker.params.LP_on = self.config["LP_on"]
		peak_picker.params.Medfilt_on = self.config["Medfilt_on"]
		peak_picker.params.Polyfit_on = self.config["Polyfit_on"]
		peak_picker.params.isMedianPositive = self.config["isMedianPositive"]

		peak_picker.params.QuadThresh_a = (100 - peak_picker.params.rawSensitivity) / 1000.0
		peak_picker.params.QuadThresh_b = 0.0
		peak_picker.params.QuadThresh_c = (100 - peak_picker.params.rawSensitivity) / 1500.0
		peak_picker.params.aCoeffs = [1.0000, -0.5949, 0.2348]
		peak_picker.params.bCoeffs = [0.1600,	 0.3200, 0.1600]
		
		# Preprocess to obtain features, times, and input boundary indeces
		F = self._preprocess()
		
		# Self similarity matrix
		S = compute_ssm(F)
		
		novelty = getNoveltyCurve(S, kernel_size=self.config["kernel_size"], normalise=False)
		smoothed_novelty, est_idxs = peak_picker.process(novelty)
		
		# Add first and last frames
		est_idxs = np.concatenate(([0], est_idxs, [F.shape[0] - 1]))
		
		# Empty labels
		est_labels = np.ones(len(est_idxs) - 1) * -1

		# Post process estimations
		est_idxs, est_labels = self._postprocess(est_idxs, est_labels)
			
		return est_idxs, est_labels
		