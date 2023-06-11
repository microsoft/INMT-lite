import json 
from utils import get_interface_mapping
import numpy as np
analysis_path = '/home/t-hdiddee/INMT-lite/user-study/data/validation_score.dsv'
with open(analysis_path,'r') as file: 
    records = file.read().strip().split('\n')
print(f'{len(records)} are the number of records being analysed.')


interfacewise_clusters = {} # interface: scores of all sentences (multiple) 
sentencewise_clusters = {} # sentence - scores of all formats

for record in records:
    try: 
        tid, scorer_id, sid, source, translation, score, mode, _ = record.split('$')
        interface = get_interface_mapping(mode)
        if sid in sentencewise_clusters:
            sentencewise_clusters[sid].append((interface, int(score)))
        else: 
            sentencewise_clusters[sid] = [(interface,int(score))]
        if interface in interfacewise_clusters:
            interfacewise_clusters[interface].append(int(score))
        else: 
            interfacewise_clusters[interface] = [int(score)]   
    except: 
        print(record)
     
# Generating sentence wise - interface wise mapping to compute the average IAA per interface

DBOW, SBOW, NWD, NWBOW, PE, B = [],[],[],[],[],[]
for sid in sentencewise_clusters:
    # print(sentencewise_clusters[sid])
    dbow, sbow, nwd, nwbow, pe, b = [],[],[],[],[],[]
    for mappings in sentencewise_clusters[sid]:
        if mappings[0] == 'NWD':
            nwd.append(mappings[1])
        if mappings[0] == 'NWBOW':
            nwbow.append(mappings[1])
        if mappings[0] == 'DBOW':
            dbow.append(mappings[1])
        if mappings[0] == 'SBOW':
            sbow.append(mappings[1])
        if mappings[0] == 'B':
            b.append(mappings[1])
        if mappings[0] == 'PE':
            pe.append(mappings[1])
            
    # print(dbow, nwbow, b, pe, sbow, nwd)
        
    # dbow.sort(reverse = True)
    # nwbow.sort(reverse = True)
    # sbow.sort(reverse = True)
    # b.sort(reverse = True)
    # pe.sort(reverse = True)
    # nwd.sort(reverse = True)
    
    dbow.sort()
    nwbow.sort()
    sbow.sort()
    b.sort()
    pe.sort()
    nwd.sort()
    # print(dbow[:3], nwbow[:3], b[:3], pe[:3], sbow[:3], nwd[:3])
    
    DBOW.append(dbow[:3])
    NWBOW.append(nwbow[:3])
    SBOW.append(sbow[:3])
    B.append(b[:3])
    PE.append(pe[:3])
    NWD.append(nwd[:3])


# print(len(DBOW), len(NWBOW), len(SBOW), len(PE), len(B), len(NWD))
# print(B)

for interface in interfacewise_clusters:
    try:
        print(f'Average sentence quality for {interface} is {np.average(interfacewise_clusters[interface])}')
    except: 
        print(interface)
        
B_STD, PE_STD, SBOW_STD, DBOW_STD, NWD_STD, NWBOW_STD = [],[],[],[],[],[]
for sentence_stats in zip(B, PE, SBOW, DBOW, NWD, NWBOW):
    B_STD.append(np.std(sentence_stats[0]))
    PE_STD.append(np.std(sentence_stats[1]))
    SBOW_STD.append(np.std(sentence_stats[2]))
    DBOW_STD.append(np.std(sentence_stats[3]))
    NWD_STD.append(np.std(sentence_stats[4]))
    NWBOW_STD.append(np.std(sentence_stats[5]))
    
print(f'Average standard deviation in interface quality assesement of B is {np.nanmean(B_STD)}')
print(f'Average standard deviation in interface quality assesement of PE is {np.nanmean(PE_STD)}')
print(f'Average standard deviation in interface quality assesement of SBOW is {np.nanmean(SBOW_STD)}')
print(f'Average standard deviation in interface quality assesement of DBOW is {np.nanmean(DBOW_STD)}')
print(f'Average standard deviation in interface quality assesement of NWD is {np.nanmean(NWD_STD)}')
print(f'Average standard deviation in interface quality assesement of NWBOW is {np.nanmean(NWBOW_STD)}')

