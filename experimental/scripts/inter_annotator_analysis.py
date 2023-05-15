import json 
import numpy as np
analysis_path = '/home/t-hdiddee/INMT-lite/experimental/data/validation_score.dsv'
with open(analysis_path,'r') as file: 
    records = file.read().split('\n')
print(f'{len(records)} are the number of records being analysed.')


## Computing the stats per interface
interface_cluster = {'B': [], 'PE': [], 'SBOW': [], 'DBOW': [], 'NWBOW':[], 'NWD': []}
translation_cluster = {}
for record in records:
    try: 
        wid, s, t, i, log = record.split('$')
    except: 
        print(s)
    if 'Next Word BOW' in i: 
        i = 'NBOW'
    if 'Next Word Dropdown' in i: 
        i = 'NWD' 
    if 'BASELINE' in i: 
        i = 'B'
    if 'DYNAMIC_BOW' in i: 
ytho    if 'STATIC_BOW' in i : 
        i = 'SBOW' 
    if 'POST_EDITED' in i : 
        i = 'PE'

    if s in translation_cluster: 
        translation_cluster[s].append([i,t])
    else: 
        translation_cluster[s] = [[i,t]] 

del translation_cluster['source']
print(len(translation_cluster.keys())) # Set of all the parallel translations 
