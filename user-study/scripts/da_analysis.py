import json 
from utils import get_interface_mapping
import numpy as np
analysis_path = '/home/t-hdiddee/ACL/user-study/translations.dsv'
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
    i = get_interface_mapping(i)
    if s in translation_cluster: 
        translation_cluster[s].append([i,t])
    else: 
        translation_cluster[s] = [[i,t]] 

del translation_cluster['source']
print(len(translation_cluster.keys())) # Set of all the parallel translations 

# Writing file-wise values + commands for BLEU computation 

for key in translation_cluster:
    # Write in all the corresponding interface files that translation was gleaned for  
    translations = translation_cluster[key]
    if len(translations) != 6: 
        print(translations)
    else: 
        for translation in translations: 
            i = translation[0]
            source_file_name =  f'./pair-wise-bleu/{i}_translations.txt'    
            with open(source_file_name, 'a') as file: 
                file.write(translation[1] + '\n')

























#     if 'BASELINE' in i: 
#         interface_cluster['B'].append((t,log))
#     elif 'POST_EDITED' in i:
#         interface_cluster['PE'].append((t,log))
#     elif 'STATIC_BOW' in i:
#         interface_cluster['SBOW'].append((t,log))
#     elif 'Next Word BOW' in i:
#         interface_cluster['NWBOW'].append((t,log))
#     elif 'DYNAMIC_BOW' in i:
#         interface_cluster['DBOW'].append((t,log))
#     elif 'Next Word Dropdown' in i:
#         interface_cluster['NWD'].append((t,log))


# for key in interface_cluster.keys():  
#     print(f'Computing Keystroke stats for {key} which has {len(interface_cluster[key])} records.')
#     time_taken, nobp, tok, total_suggestions, tapped_suggestions = [], 0, 0, 1, 0 # 1 to avoid underflow for the first 2 interfaces        
#     for record in interface_cluster[key]:
#         translation, logs = record[0], record[1]
#         logs = logs.replace('""','"')
#         logs = logs.strip('"')
#         logs = json.loads(logs)
        
#         # Compute the total time taken to complete the translations 
#         for ks, ts in enumerate(logs): 
#             if 'button' in ts['message']:
#                 time_taken.append(ts['message']['time_taken']//6000)
    
#         # Compute the total number of keystrokes per interface 
#         tok = len(logs)
#         # Compute the number of backspaces per intefaces 
#         for ks, ts in enumerate(logs): 
#             for elements in ts['message']: 
#                 try: 
#                     if ts['message'][elements] == 'bp':
#                         nobp += 1 
#                 except: 
#                     continue
        
#         # Compute the total number of tapped idx
#         if key not in ['B', 'PE']:
#             for ts in logs: 
#                 # logging all the times suggestions were generated: 
#                 message = ts['message']
#                 try: 
#                     for outer_seed in message: 
#                         # print(message[key])
#                         for seed in message[outer_seed]:
#                             if seed == 'total_invocation_time': 
#                                 total_suggestions += 1 
#                                 # print(message[outer_seed])
#                 except: 
#                     if 'type' in message: 
#                         tapped_suggestions += 1 

#     # Printing Interface-Wide Stats 
#     print(f'Found valid records for {len(time_taken)} records.')
#     print(f'For interface {key}: the average time taken to complete a translation is {np.average(time_taken)}')
#     print(f'For interface {key}: the average number of backspaces is is {nobp/len(time_taken)}')
#     print(f'For interface {key}: the average number of keystrokes is {tok/len(time_taken)}')
#     ### Specific computation for SBOW which does not have the total number of suggestions shown
#     if key == 'SBOW':
#         total_suggestions = 0 
#         with open('/home/t-hdiddee/INMT-lite/experimental/scripts/final_study_dump/user-study-all-task-dump.json') as file:
#             samples = file.read().split('\n')
#             for sample in samples:
#                 record = json.loads(sample)
#                 total_suggestions += len(record['BOW'])    

#     print(f'For interface {key}: out of {total_suggestions} only {tapped_suggestions} were used and average usage {tapped_suggestions/total_suggestions}')
#     print('**************************************************************')

#     with open('analysis_stats.json', 'a') as f:
#         obj = {key: [np.average(time_taken), nobp/len(time_taken), tok/len(time_taken), (tapped_suggestions/total_suggestions)*100]}
#         f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    



















































