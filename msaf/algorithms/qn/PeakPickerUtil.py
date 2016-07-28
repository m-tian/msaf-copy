#!/usr/bin/env python
# encoding: utf-8
"""
PeakPickerUtil.py

Created by george fazekas and mi tian on 2015-05-22.
Copyright (c) 2015 Queen Mary University of London. All rights reserved.

This utility is adapted from an onset detection algorithm from the Queen Mary DSP library.
For details see:
https://code.soundsoftware.ac.uk/hg/qm-dsp
https://code.soundsoftware.ac.uk/projects/qm-dsp/repository/entry/dsp/onsets/PeakPicking.cpp

"""

import numpy as np
from numpy import *
import sys
from scipy.signal import medfilt, filtfilt, butter

from msaf.algorithms.interface import SegmenterInterface

class PeakPicker():
	'''Separate Peak Picker implementation that can be used in all plugins identically.'''

	def __init__(self):
		'''Initialise the PeakPicker, but for now, we just define a class for the parameters.'''
		class Params(object):
			'''Just create a small efficient object for storing the parameters.'''
			__slots__ = ['alpha','delta','QuadThresh_a','QuadThresh_b','QuadThresh_c','aCoeffs','bCoeffs',\
			'preWin','postWin','LP_on','Medfilt_on','Polyfit_on','isMedianPositive','rawSensitivity']
		self.params = Params()

		
	def process(self, onset_df, calcu_env=False, prune = False):
		'''Smooth and peak pick the detection function.'''
		
		smoothed_df = self.smooth_df(onset_df)		
		onsets = self.quadEval(smoothed_df, prune)
		
		return smoothed_df, onsets
	
	def envFollower(self, sig):
		'''Peak position of the signal envelope.'''
		env = []
		if not len(sig): return env
		i = 1
		while i < len(sig):
			pos = 1
			while  (i+pos) < len(sig):
				if sig[i+pos] < sig[i+pos-1]: break
				pos += 1
			if sig[i+pos-1] > sig[i+pos-2]: env.append(i+pos-1) 
			i += pos	
		
		env = list(sort(env))
		
		if not len(env): return env
		
		if env[-1] == len(sig):
			env = env[:-1]
		return env
		
	def quadEval(self, smoothed_df, prune=False):
		'''Assess onset locations using the paramaters of a quadratic fucntion (or simple thresholding).'''
		onsets = []
		x_array = np.array(xrange(-2,3),dtype=np.float32)
		max_index = []
		
		# if prune:
		# 	smoothed_df = [abs(i) for i in smoothed_df]
		# find maxima in the smoothed function, NOTE: could do this later with scipy.signal.argrelmax (using ver > .0.11)
		for i in xrange(2,len(smoothed_df)-2) :
			if smoothed_df[i] > smoothed_df[i-1] and smoothed_df[i] > smoothed_df[i+1] and smoothed_df[i] > 0 :
				max_index.append(i)
		# in case the last local maxima with an incomplete peak shape is missed
		last = len(smoothed_df)-1
		if smoothed_df[last] >= max(smoothed_df[i] for i in xrange(last-2, last-1)):
			max_index.append(last)

		# if len(max_index) == 0 :
		# 	return onsets
		
		# if the polynomial fitting is not on, just return the detected peaks above a threshold 
		# calculated using 100-rawSensitivity value considering the smallest and largest peaks
		if not self.params.Polyfit_on : 
			if not max_index :
				return onsets
			max_index = np.array(max_index)
			smoothed_df = np.array(smoothed_df)
			smoothed_df_peaks = smoothed_df[max_index]
			min_df, max_df = smoothed_df_peaks.min(), smoothed_df_peaks.max()
			range_df = max_df-min_df
			sensitivity = (100-self.params.rawSensitivity) / 100.0
			threshold = min_df + sensitivity * range_df
			return max_index[smoothed_df[max_index]>=threshold]

		# NOTE: GF: The order of the polynomial coefficients is reversed in the C++ implementation!
		# But it is numerically equivalent and accurate (checked by printing the results from the C++ code).
		for j in xrange(len(max_index)) :
			if max_index[j] + 2 > len(smoothed_df) :
				onsets.append(max_index[j])
			else :
				y_array = list()
				for k in xrange(-2,3) :
					selMax = smoothed_df[max_index[j] + k]
					y_array.append(selMax)
				coeffs = polyfit(x_array,np.array(y_array),2)
				# print coeffs

				if coeffs[0] < -self.params.QuadThresh_a or coeffs[2] > self.params.QuadThresh_c :
					onsets.append(max_index[j])
					# print max_index[j]
		
		# If the arg prune is on, remove onset candidates that have spurious peaks on its both side neighbourhood (1.5-2s)
		if prune :
			remove = []
			step = 50
			onsets.sort()
			for idx in xrange(1, len(onsets) - 1):	
				if onsets[idx+1] - onsets[idx] < step and onsets[idx] - onsets[idx-1] < step :
					remove.append(onsets[idx])
			onsets = [i for i in onsets if not i in remove]
			print 'remove', remove, onsets
		return onsets
		

	def smooth_df(self, onset_df):
		'''Smooth the detection function by 1) removing DC and normalising, 2) zero-phase low-pass filtering,
		and 3) adaptive thresholding using a moving median filter with separately configurable pre/post window sizes.
		'''
		
		out_df = self.removeDCNormalize(onset_df)

		if self.params.LP_on :
			# Now we get the exact same filtered function produced by the QM-Vamp plugin:
			out_df = filtfilt(self.params.bCoeffs, self.params.aCoeffs, out_df)
			
		if self.params.Medfilt_on :
			out_df = self.movingMedianFilter(out_df)

		return out_df

		
		
	def movingMedianFilter(self, onset_df):
		'''Adaptive thresholding implementation using a moving median filter with configurable pre/post windows. '''
		# TODO: Simplify and vectorise this where appropriate, may replace While loops with For if faster, 
		# perhaps use C extension module, Theano or similar...
		length = len(onset_df)
		isMedianPositive = self.params.isMedianPositive
		preWin = int(self.params.preWin) 
		postWin = int(self.params.postWin)
		index = 0
		delta = self.params.delta / 10.0
		y = np.zeros(postWin+preWin+1, dtype=np.float64)
		scratch = np.zeros(length, dtype=np.float64)
		output = np.zeros(length, dtype=np.float64)
		
		for i in xrange(preWin) :
			if index >= length : break
			k = i + postWin + 1;
			for j in xrange(k) :
				if j < length: y[j] = onset_df[j]
			scratch[index] = np.median(y[:k])
			index += 1
			
		i = 0
		while True :
			if i+preWin+postWin >= length : break
			if index >= length : break
			l = 0
			j = i
			while True:
				if j >= ( i+preWin+postWin+1) : break
				y[l] = onset_df[j]
				l += 1
				j += 1
			i += 1
			scratch[index] = np.median(y[:preWin+postWin+1])
			index += 1

		i = max(length-postWin, 1)
		while True :
			if i >= length : break
			if index >= length : break
			k = max(i-preWin, 1)
			l = 0
			j = k
			while True :
				if j >= length : break
				y[l] = onset_df[j]
				j += 1
				l += 1
			scratch[index] = np.median(y[:l])
			index += 1
			i += 1
			
		for i in xrange(length) :
			value = onset_df[i] - scratch[i] - delta
			output[i] = value
			if isMedianPositive and value < 0.0 :
				output[i] = 0.0

		return output.tolist()
		
	def removeDCNormalize(self,onset_df):
		'''Remove constant offset (DC) and regularise the scale of the detection function.'''
		DFmin,DFmax = self.getFrameMinMax(onset_df)
		DFAlphaNorm = self.getAlphaNorm(onset_df,self.params.alpha) 
		for i,v in enumerate(onset_df) :
			onset_df[i] = (v - DFmin) / DFAlphaNorm
		return onset_df
		
	def getAlphaNorm(self, onset_df, alpha):
		'''Calculate the alpha norm of the detecion function'''
		# TODO: Simplify or vectorise this.
		# a = 0.0
		# for i in onset_df :
		# 	a += pow(fabs(i), alpha)
		a = sum( np.power(fabs(onset_df), alpha) )
		a /= len(onset_df)
		a = pow(a, (1.0 / alpha))
		return a
		
	def getFrameMinMax(self, onset_df):
		'''Just return the min/max of the detecion function'''
		return min(onset_df),max(onset_df)
