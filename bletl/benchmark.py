"""Contains benchmarking code"""
import pathlib
import tqdm

import bletl

fp1 = pathlib.Path('data', 'BL1', 'NT_1400_BS20BS10_15min_20160222_151645.csv')
fppro = pathlib.Path('data', 'BLPro', '18-FZJ-Test2--2018-02-07-10-01-11.csv')



for i in tqdm.trange(42):
    bld1 = bletl.parse(fp1)
    bldp = bletl.parse(fppro)


