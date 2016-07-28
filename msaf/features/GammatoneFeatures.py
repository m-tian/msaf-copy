#!/usr/bin/env python
# encoding: utf-8
"""
GammatoneFeatures.py

Calculate gammatone features from the audio input.

"""

import sys
import os
import librosa
import numpy as np
import scipy as sp

DEFAULT_FILTER_NUM = 64
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100/2

class ERB(object):
	'''Computes the ERB filterbank and the gammatone filter.'''
	
	def __init__(self):
		self.inputSampleRate = 44100
		self.stepSize = 512
		self.blockSize = 1024
		self.nfft = self.blockSize
		self.lowFreq = DEFAULT_LOW_FREQ
		self.highFreq = self.inputSampleRate / 2
		self.nBands = DEFAULT_FILTER_NUM
		self.nChannels = 1
		self.useLog = True
		self.subbands = [0, 4, 7, 10, 12, 15, 18, 21, 25, 29, 37, 45, 63] #To be discussed
		self.nSubbands = 12
		self.nCepstral = 20
		self.gammatoneLen = 1024
		self.updated = False
				
	
	def erbFreq(self, bands):
		'''computes an array of ``num`` frequencies uniformly spaced between ``high_freq`` and ``low_freq`` on an ERB scale.
		For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
		"Suggested formulae for calculating auditory-filter bandwidths and excitation patterns," 
		J. Acoust. Soc. Am. 74, 750-753.
	
		http://www.acousticscale.org/wiki/index.php/The_Gammatone_Auditory_Filterbank
		'''
	
		'''Center frequencies.
		20 channel 100 Hz ~ 44100/2 Hz:
		[  100.		   ,	177.16250825,	 272.43163729,	  390.05620853,
				  535.28205345,	   714.58596766,	935.96456368,	1209.29090453,
				 1546.75482359,	  1963.40656784,   2477.82801018,	3112.96136556,
				 3897.13236992,	  4865.31355405,   6060.68395168,	7536.55480206,
				 9358.74712947,	 11608.5272354 ,  14386.23102049,  17815.73877432]
		64 channel erb:
		[	100.		,	 122.39041981,	  146.30541606,	   171.84879804,
				  199.13144347,	   228.27177977,	259.39629818,	 292.64010278,
				  328.147497  ,	   366.07260993,	406.58006541,	 449.84569663,
				  496.05730934,	   545.4154971 ,	598.134512	,	 654.44319468,
				  714.58596766,	   778.82389633,	847.43582219,	 920.71957318,
				  998.99325656,	  1082.59663965,   1171.89262475,	1267.26882438,
				 1369.13924381,	  1477.94607816,   1594.16163189,	1718.29036892,
				 1850.87110241,	  1992.47933362,   2143.72975	,	2305.27889346,
				 2477.82801018,	  2662.12609462,   2858.97314068,	3069.22361431,
				 3293.79016254,	  3533.64757504,   3789.83701549,	4063.47054097,
				 4355.73592918,	  4667.90183427,   5001.32329378,	5357.44761052,
				 5737.82063499,	  6144.09347551,   6578.02966534,	7041.51281768,
				 7536.55480206,	  8065.30447734,   8630.05701941,	9233.26388399,
				 9877.54344782,	 10565.69237441,  11300.69775366,  12085.75006816,
				12924.25704226,	 13819.85843418,  14776.4418354 ,  15798.15954567,
				16889.44659724,	 18055.04000621,  19299.99933482,  20629.72865376]
		
	
		'''
		ear_q = 9.26449 # Glasberg and Moore Parameters
		min_bw = 24.7
		order = 1
		erb = (-ear_q * min_bw + np.exp(bands * (-np.log(self.highFreq + ear_q*min_bw) + np.log(self.lowFreq + ear_q*min_bw))) * \
		(self.highFreq + ear_q*min_bw))

		return erb 

	def make_erb_filters(self, fs, centre_freqs, width=1.0):
	    """
	    This function computes the filter coefficients for a bank of 
	    Gammatone filters. These filters were defined by Patterson and Holdworth for
	    simulating the cochlea. 
    
	    The result is returned as a :class:`ERBCoeffArray`. Each row of the
	    filter arrays contains the coefficients for four second order filters. The
	    transfer function for these four filters share the same denominator (poles)
	    but have different numerators (zeros). All of these coefficients are
	    assembled into one vector that the ERBFilterBank can take apart to implement
	    the filter.
    
	    The filter bank contains "numChannels" channels that extend from
	    half the sampling rate (fs) to "lowFreq". Alternatively, if the numChannels
	    input argument is a vector, then the values of this vector are taken to be
	    the center frequency of each desired filter. (The lowFreq argument is
	    ignored in this case.)
    
	    Note this implementation fixes a problem in the original code by
	    computing four separate second order filters. This avoids a big problem with
	    round off errors in cases of very small cfs (100Hz) and large sample rates
	    (44kHz). The problem is caused by roundoff error when a number of poles are
	    combined, all very close to the unit circle. Small errors in the eigth order
	    coefficient, are multiplied when the eigth root is taken to give the pole
	    location. These small errors lead to poles outside the unit circle and
	    instability. Thanks to Julius Smith for leading me to the proper
	    explanation.
    
	    Execute the following code to evaluate the frequency response of a 10
	    channel filterbank::
    
	        fcoefs = MakeERBFilters(16000,10,100);
	        y = ERBFilterBank([1 zeros(1,511)], fcoefs);
	        resp = 20*log10(abs(fft(y')));
	        freqScale = (0:511)/512*16000;
	        semilogx(freqScale(1:255),resp(1:255,:));
	        axis([100 16000 -60 0])
	        xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');
    
	    | Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
	    | (c) 1998 Interval Research Corporation
	    |
	    | (c) 2012 Jason Heeris (Python implementation)
	    """
	    T = 1/fs
	    # Change the followFreqing three parameters if you wish to use a different
	    # ERB scale. Must change in ERBSpace too.
	    # TODO: factor these out
	    ear_q = 9.26449 # Glasberg and Moore Parameters
	    min_bw = 24.7
	    order = 1

	    erb = width*((centre_freqs/ear_q)**order + min_bw**order)**(1/order)
	    B = 1.019*2*np.pi*erb

	    arg = 2*centre_freqs*np.pi*T
	    vec = np.exp(2j*arg)

	    A0 = T
	    A2 = 0
	    B0 = 1
	    B1 = -2*np.cos(arg)/np.exp(B*T)
	    B2 = np.exp(-2*B*T)
    
	    rt_pos = np.sqrt(3 + 2**1.5)
	    rt_neg = np.sqrt(3 - 2**1.5)
    
	    common = -T * np.exp(-(B * T))
    
	    # TODO: This could be simplified to a matrix calculation involving the
	    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
	    # terms
	    k11 = np.cos(arg) + rt_pos * np.sin(arg)
	    k12 = np.cos(arg) - rt_pos * np.sin(arg)
	    k13 = np.cos(arg) + rt_neg * np.sin(arg)
	    k14 = np.cos(arg) - rt_neg * np.sin(arg)

	    A11 = common * k11
	    A12 = common * k12
	    A13 = common * k13
	    A14 = common * k14

	    gain_arg = np.exp(1j * arg - B * T)

	    gain = np.abs(
	            (vec - gain_arg * k11)
	          * (vec - gain_arg * k12)
	          * (vec - gain_arg * k13)
	          * (vec - gain_arg * k14)
	          * (  T * np.exp(B*T)
	             / (-1 / np.exp(B*T) + 1 + vec * (1 - np.exp(B*T)))
	            )**4
	        )

	    allfilts = np.ones_like(centre_freqs)
    
	    fcoefs = np.column_stack([
	        A0*allfilts, A11, A12, A13, A14, A2*allfilts,
	        B0*allfilts, B1, B2,
	        gain
	    ])
    
	    return fcoefs


	def erb_filterbank(self, wave, coefs):
	    """
	    :param wave: input data (one dimensional sequence)
	    :param coefs: gammatone filter coefficients
    
	    Process an input waveform with a gammatone filter bank. This function takes
	    a single sound vector, and returns an array of filter outputs, one channel
	    per row.
    
	    The fcoefs parameter, which completely specifies the Gammatone filterbank,
	    should be designed with the :func:`make_erb_filters` function.
    
	    | Malcolm Slaney @ Interval, June 11, 1998.
	    | (c) 1998 Interval Research Corporation
	    | Thanks to Alain de Cheveigne' for his suggestions and improvements.
	    |
	    | (c) 2013 Jason Heeris (Python implementation)
	    """
	    output = np.zeros((coefs[:,9].shape[0], wave.shape[0]))
    
	    gain = coefs[:, 9]
	    # A0, A11, A2
	    As1 = coefs[:, (0, 1, 5)]
	    # A0, A12, A2
	    As2 = coefs[:, (0, 2, 5)]
	    # A0, A13, A2
	    As3 = coefs[:, (0, 3, 5)]
	    # A0, A14, A2
	    As4 = coefs[:, (0, 4, 5)]
	    # B0, B1, B2
	    Bs = coefs[:, 6:9]
    
	    # Loop over channels
	    for idx in range(0, coefs.shape[0]):
	        # These seem to be reversed (in the sense of A/B order), but that's what
	        # the original code did...
	        # Replacing these with polynomial multiplications reduces both accuracy
	        # and speed.
	        y1 = sgn.lfilter(As1[idx], Bs[idx], wave)
	        y2 = sgn.lfilter(As2[idx], Bs[idx], y1)
	        y3 = sgn.lfilter(As3[idx], Bs[idx], y2)
	        y4 = sgn.lfilter(As4[idx], Bs[idx], y3)
	        output[idx, :] = y4/gain[idx]
        
	    return output


	def getFFTWeights(self, nfft, fs, channels, width, fmin, fmax, maxlen):
		'''Calculate a spectrogram-like time frequency magnitude array based on
		an FFT-based approximation to gammatone subband filters. This function is taken from 
		the gammatone toolkit by Jason Heeris.'''
		weights, _ = fft_weights(nfft, fs, channels, width, fmin, fmax, maxlen)
		return weights
	

	def fft_weights(self, nfft, fs, nfilts, width, fmin, fmax, maxlen):
		"""
		:param nfft: the source FFT size
		:param sr: sampling rate (Hz)
		:param nfilts: the number of output bands required (default 64)
		:param width: the constant width of each band in Bark (default 1)
		:param fmin: lower limit of frequencies (Hz)
		:param fmax: upper limit of frequencies (Hz)
		:param maxlen: number of bins to truncate the rows to
	
		:return: a tuple `weights`, `gain` with the calculated weight matrices and
				 gain vectors
	
		Generate a matrix of weights to combine FFT bins into Gammatone bins.
	
		Note about `maxlen` parameter: While wts has nfft columns, the second half
		are all zero. Hence, aud spectrum is::
	
			fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft))
	
		`maxlen` truncates the rows to this many bins.
	
		| (c) 2004-2009 Dan Ellis dpwe@ee.columbia.edu	based on rastamat/audspec.m
		| (c) 2012 Jason Heeris (Python implementation)
		"""
		ucirc = np.exp(1j * 2 * np.pi * np.arange(0, nfft/2 + 1)/nfft)[None, ...]
	
		# Common ERB filter code factored out
		cf_array = self.erbFreq(np.arange(1, self.nBands+1)/float(self.nBands))[::-1]

		_, A11, A12, A13, A14, _, _, _, B2, gain = (self.make_erb_filters(fs, cf_array, width).T
		)
	
		A11, A12, A13, A14 = A11[..., None], A12[..., None], A13[..., None], A14[..., None]

		r = np.sqrt(B2)
		theta = 2 * np.pi * cf_array / fs	 
		pole = (r * np.exp(1j * theta))[..., None]
	
		GTord = 4
	
		weights = np.zeros((nfilts, nfft))

		weights[:, 0:ucirc.shape[1]] = (
			  np.abs(ucirc + A11 * fs) * np.abs(ucirc + A12 * fs)
			* np.abs(ucirc + A13 * fs) * np.abs(ucirc + A14 * fs)
			* np.abs(fs * (pole - ucirc) * (pole.conj() - ucirc)) ** (-GTord)
			/ gain[..., None]
		)

		weights = weights[:, 0:maxlen]

		return weights, gain

	
	def get_stft(self, x, nfft, fs, block_size, step_size):
		'''Compute STFT of input chunked audio data.'''
		windowed = x * np.hanning(nfft)
		padded = np.append(windowed, np.zeros(nfft)) # add 0s to double the length of the data
		stft = np.fft.fft(windowed)[:nfft/2+1] / nfft # take the Fourier Transform and scale by the number of samples
		
		return stft
	
	def fft_gtgram(self, wave, fs, window_time, hop_time, channels, f_min):
		"""
		Calculate a spectrogram-like time frequency magnitude array based on
		an FFT-based approximation to gammatone subband filters.

		A matrix of weightings is calculated (using :func:`gtgram.fft_weights`), and
		applied to the FFT of the input signal (``wave``, using sample rate ``fs``).
		The result is an approximation of full filtering using an ERB gammatone
		filterbank (as per :func:`gtgram.gtgram`).

		``f_min`` determines the frequency cutoff for the corresponding gammatone
		filterbank. ``window_time`` and ``hop_time`` (both in seconds) are the size
		and overlap of the spectrogram columns.

		| 2009-02-23 Dan Ellis dpwe@ee.columbia.edu
		|
		| (c) 2013 Jason Heeris (Python implementation)
		"""
		width = 1 # Was a parameter in the MATLAB code

		nfft = int(2**(np.ceil(np.log2(2 * window_time * fs))))

		gt_weights, _ = self.fft_weights(nfft, fs, self.nBands, 1, f_min, fs/2, nfft/2 + 1)

		stft = self.get_stft(wave, nfft, fs, nwin, nhop)

		gtgram = gt_weights.dot(np.abs(stft)) / nfft
		
		return gtgram


class GTfeatures(ERB):
	

	def getDCTMatrix(self, size):
		'''Calculate the square DCT transform matrix. '''
		DCTmx = np.array(xrange(size), np.float64).repeat(size).reshape(size,size)
		DCTmxT = np.pi * (DCTmx.transpose()+0.5) / size
		DCTmxT = (1.0 / np.sqrt( size / 2.0)) * np.cos(DCTmx * DCTmxT)
		DCTmxT[0] = DCTmxT[0] * (np.sqrt(2.0)/2.0)
		
		self.updated = True
		
		return DCTmxT

	def getDCT(self, data, nBins=64):
		'''Compute DCT of input data.'''
		dctMatrix = self.getDCTMatrix(size=data.shape[1])
		dct = np.dot(data, dctMatrix)
	
		return dct


	def getPrincipalComponents(self, data, N):
		'''Get the first N compoments of the data after a PCA.'''
		pca = PCA(n_components=N)
		pca.fit(data)

		return pca.transform(data)
	
	def getSubbandContrast(self, data, subbands):
		'''Get the contrast within each subband of the data.
		'''
		N = len(subbands)
		contrast = np.zeros((data.shape[0], N-1))

		for i in xrange(1, N):
			contrast[:, i-1] = np.max(data[:,subbands[i-1]:subbands[i]], axis=-1) - np.min(data[:,subbands[i-1]:subbands[i]], axis=-1)

		return contrast

	def STFT(self, x, nfft, fs, block_size, step_size):
		'''Compute STFT of input chunked audio data.'''
		windowed = x * np.hanning(nfft)
		padded = np.append(windowed, np.zeros(nfft)) # add 0s to double the length of the data
		stft = np.fft.fft(windowed)[:nfft/2+1] / nfft # take the Fourier Transform and scale by the number of samples

		return stft	

	def getGCC(self, gtgram, nCoeff=20, useLog=True):
		'''Computes the gammatonegram cepstral coefficents feature.'''
	
		if useLog:
			gtgram = np.log(gtgram + 1e-5)
			gtgram[np.isinf(gtgram)] = 0.0
			gtgram[np.isnan(gtgram)] = 0.0
		
		gcc = self.getDCT(gtgram)[:, :self.nCepstral]
	
		return gcc
	
	def getGPC(self, gtgram, nComp=6, useLog=True):
		'''Computes the gammatonegram principal components feature.'''
	
		if useLog:
			gtgram = np.log(gtgram + 1e-5)
			gtgram[np.isinf(gtgram)] = 0.0
			gtgram[np.isnan(gtgram)] = 0.0
	
		return self.getPrincipalComponents(gtgram, N=nComp)

	def getGSC(self, gtgram, nBands=6, useLog=True):
		'''Computes the gammatonegram spectral constrast feature.'''
	
		if useLog:
			gtgram = np.log(gtgram + 1e-5)
			gtgram[np.isinf(gtgram)] = 0.0
			gtgram[np.isnan(gtgram)] = 0.0
	
		return self.getSubbandContrast(gtgram, nBands)
	
def main():
	
	gt = GTfeatures()

	sr, y = sp.io.wavfile.read('/Users/mitian/Documents/audio/OnsetDB/bello/audio2/Jaillet15.wav')
	
	# Use mean of two channels. 
	if len(y.shape) > 1:
		y = np.mean(v, axis=-1)
	
	# Process data by chunks.
	nFrames = y.shape[0]
	window_size, hope_size = 1024, 512
	nStep = (nFrames - window_size) / hope_size + 1
	print 'nFrames', nFrames, nStep
	
	for i in xrange(nStep):
		wav = y[i*hope_size: i*hope_size + window_size]
		gtStripe = gt.fft_gtgram(wav, sr, 1024, 512, 1, 100)
		gammatoneGram.append(gtStripe)
		print i, gtStripe.shape
	gammatoneGram = np.hstack(gammatoneGram)
	
	print 'gammatoneGram', gammatoneGram.shape
	
	gcc = gt.getGCC(gammatoneGram)
	
	print gcc.shape

if __name__ == '__main__':
	main()

