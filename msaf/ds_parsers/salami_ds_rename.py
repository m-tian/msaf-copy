#!/usr/bin/env python
# encoding: utf-8
"""
salami_ds_rename.py

Created by mi tian on 2015-05-27.
Copyright (c) 2015 __MyCompanyName__. All rights reserved.
"""

import sys
import os
from os import rename, listdir
from os.path import splitext, basename, dirname, join
	
def main():
	inDir = ('/Users/mitian/Documents/experiments/msaf-devel/datasets/InternetArchive/references')
	fnames = listdir(inDir)
	
	for f in fnames:
		f_name, f_ext = splitext(f)
		f_name = f_name.replace(f_name[:7],'')
		rename(join(inDir, f), join(inDir, f_name+f_ext))



if __name__ == '__main__':
	main()

