import json 
import numpy as np
from utils import get_interface_mapping

analysis_path = '/home/t-hdiddee/INMT-lite/user-study/raw-responses/translations.dsv'

with open(analysis_path,'r') as file: 
    records = file.read().strip().split('\n')
print(f'{len(records)} are the number of records being analysed.')


## Computing the stats per interface
interface_cluster = {'B': [], 'PE': [], 'SBOW': [], 'DBOW': [], 'NWBOW':[], 'NWD': []}
for record in records:
    try: 
        wid, s, t, i, log = record.split('$')
        interface = get_interface_mapping(i)
        if interface is None: 
            continue
    except: 
        print(s)
    interface_cluster[interface].append((t,log))

for key in interface_cluster.keys():  
    print(f'Computing Keystroke stats for {key} which has {len(interface_cluster[key])} records.')
    time_taken, nobp, tok, total_suggestions, tapped_suggestions, tidx = [], 0, 0, 1, 0, 0 # 1 to avoid underflow for the first 2 interfaces        
    for record in interface_cluster[key]:
        translation, logs = record[0], record[1]
        logs = logs.replace('""','"')
        logs = logs.strip('"')
        logs = json.loads(logs)
    
        
        # Compute the total time taken to complete the translations 
        for ks, ts in enumerate(logs): 
            if 'button' in ts['message']:
                time_taken.append(ts['message']['time_taken']/60000)
                # time_taken = ts['message']['time_taken']//60000
        
        # Compute the total number of keystrokes per interface 
        tok += len(logs) #Number of keystrokes per record 
        # Compute the number of backspaces per intefaces 
        for ks, ts in enumerate(logs): 
            for elements in ts['message']: 
                try: 
                    if ts['message'][elements] == 'bp':
                        nobp += 1 
                except: 
                    continue
                
    # Printing Interface-Wide Stats 
    print(f'Found valid records for {len(time_taken)} records.')
    print(f'For interface {key}: the average time taken to complete a translation is {np.average(time_taken)}')
    print(f'For interface {key}: the average number of backspaces is {nobp/len(time_taken)}')
    print(f'For interface {key}: the average number of keystrokes is {tok/len(time_taken)}')


for key in interface_cluster.keys():  
    if key not in ['B','PE']:
        print(f'Computing Keystroke stats for {key} which has {len(interface_cluster[key])} records.')
        
        total_suggestions, tapped_suggestions, tidx = 0, 0, 0 
        for record in interface_cluster[key]:
            translation, logs = record[0], record[1]
            logs = logs.replace('""','"')
            logs = logs.strip('"')
            logs = json.loads(logs)
            
            
            for ts in logs: 
                # logging all the times suggestions were generated: 
                message = ts['message']
                if 'tapped_idx' in message: 
                        tapped_suggestions += 1
                try:
                    for outer_seed in message: 
                        for seed in message[outer_seed]:
                            if seed == 'total_invocation_time': 
                                total_suggestions += 1
                except Exception as e: 
                    pass         
                    
                if key == 'SBOW':     ## Specific computation for SBOW which does not have the total number of suggestions shown
                    total_suggestions = 0 
                    with open('/home/t-hdiddee/INMT-lite/user-study/data/final_study_dump/user-study-all-task-dump.json') as file:
                        samples = file.read().split('\n')
                        for sample in samples:
                            record = json.loads(sample)
                            total_suggestions += len(record['BOW'])    
                    
        print(f'{total_suggestions} are total suggestions.')
        # print(f'{tidx} are tapped indices from earlier method.')
        print(f'{tapped_suggestions} are tapped indices from new method.')
        # print(f'For interface {key}: the average number of tapped indices by old method {(tidx/total_suggestions)*100}')
        print(f'For interface {key}: the average number of tapped indices by new method {(tapped_suggestions/total_suggestions)}')
        print('*******************************************************************************************************************')
    else: 
        print('Not Applicable for this interface.')            

    # with open('analysis_stats.json', 'a') as f:
    #     obj = {key: [np.average(time_taken), nobp/len(time_taken), tok/len(time_taken), (tapped_suggestions/total_suggestions)]}
    #     f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    


















































