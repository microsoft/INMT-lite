import json 
import numpy as np
analysis_path = '/home/t-hdiddee/INMT-lite/user-study/data/validation_score.dsv'
with open(analysis_path,'r') as file: 
    records = file.read().strip().split('\n')
print(f'{len(records)} are the number of records being analysed.')


interannotator_clusters = {} #sentence - scores of all formats 
for record in records:
    try: 
        tid, scorer_id, sid, source, translation, score, mode = record.split('$')
    except: 
        print(sid)
    if sid in interannotator_clusters:
        interannotator_clusters[sid].append([mode, score])    
    