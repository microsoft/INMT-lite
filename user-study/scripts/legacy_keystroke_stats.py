import json 
import numpy as np
import matplotlib.pyplot as plt
analysis_path = '/home/t-hdiddee/ACL/user-study/translations.dsv'
with open(analysis_path,'r') as file: 
    records = file.read().split('\n')
print(f'{len(records)} are the number of records being analysed.')


## Computing the stats per interface
interface_cluster = {'B': [], 'PE': [], 'SBOW': [], 'DBOW': [], 'NWBOW':[], 'NWD': []}
for record in records:
    try: 
        wid, s, t, i, log = record.split('$')
    except: 
        print(s)
    if 'BASELINE' in i: 
        interface_cluster['B'].append((t,log))
    elif 'POST_EDITED' in i:
        interface_cluster['PE'].append((t,log))
    elif 'STATIC_BOW' in i:
        interface_cluster['SBOW'].append((t,log))
    elif 'Next Word BOW' in i:
        interface_cluster['NWBOW'].append((t,log))
    elif 'DYNAMIC_BOW' in i:
        interface_cluster['DBOW'].append((t,log))
    elif 'Next Word Dropdown' in i:
        interface_cluster['NWD'].append((t,log))


for key in interface_cluster.keys():  
    print(f'Computing Keystroke stats for {key} which has {len(interface_cluster[key])} records.')
    time_taken, nobp, tok, total_suggestions, tapped_suggestions = [], 0, 0, 1, 0 # 1 to avoid underflow for the first 2 interfaces        
    for record in interface_cluster[key]:
        translation, logs = record[0], record[1]
        logs = logs.replace('""','"')
        logs = logs.strip('"')
        logs = json.loads(logs)
        
        # Compute the total time taken to complete the translations 
        for ks, ts in enumerate(logs): 
            if 'button' in ts['message']:
                time_taken.append(ts['message']['time_taken']//60000) # minutes 
    
        # Compute the total number of keystrokes per interface 
        tok = len(logs)
        # Compute the number of backspaces per intefaces 
        for ks, ts in enumerate(logs): 
            for elements in ts['message']: 
                try: 
                    if ts['message'][elements] == 'bp':
                        nobp += 1 
                except: 
                    continue
        
        # Compute the total number of tapped idx
        if key not in ['B', 'PE']:
            for ts in logs: 
                # logging all the times suggestions were generated: 
                message = ts['message']
                try: 
                    for outer_seed in message: 
                        # print(message[key])
                        for seed in message[outer_seed]:
                            if seed == 'total_invocation_time': 
                                total_suggestions += 1 
                                # print(message[outer_seed])
                except: 
                    if 'type' in message: 
                        tapped_suggestions += 1 

    # Printing Interface-Wide Stats 
    print(f'Found valid records for {len(time_taken)} records.')
    print(f'For interface {key}: the average time taken to complete a translation is {np.average(time_taken)}')
    print(f'For interface {key}: the average number of backspaces is is {nobp/len(time_taken)}')
    print(f'For interface {key}: the average number of keystrokes is {tok/len(time_taken)}')
    ### Specific computation for SBOW which does not have the total number of suggestions shown
    if key == 'SBOW':
        total_suggestions = 0 
        with open('/home/t-hdiddee/INMT-lite/experimental/scripts/final_study_dump/user-study-all-task-dump.json') as file:
            samples = file.read().split('\n')
            for sample in samples:
                record = json.loads(sample)
                total_suggestions += len(record['BOW'])    

    print(f'For interface {key}: out of {total_suggestions} only {tapped_suggestions} were used and average usage {tapped_suggestions/total_suggestions}')
    print('**************************************************************')

    # with open('analysis_stats.json', 'a') as f:
    #     obj = {key: [np.average(time_taken), nobp/len(time_taken), tok/len(time_taken), (tapped_suggestions/total_suggestions)*100]}
    #     f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    





















































'''
इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।
ईदेन नवा तोडी मुलायम अन चमकदार बने माम 
"[{""ts"":""2023-02-23T12:51:58.492Z"",""message"":""microtask setup complete""},

{""ts"":""2023-02-23T12:53:28.673Z"",""message"":{""1677156808673"":""ई""}},{""ts"":""2023-02-23T12:53:33.055Z"",""message"":{""1677156813055"":""द""}},{""ts"":""2023-02-23T12:53:33.963Z"",""message"":{""1677156813962"":""े""}},{""ts"":""2023-02-23T12:53:34.782Z"",""message"":{""1677156814781"":""न""}},{""ts"":""2023-02-23T12:53:38.498Z"",""message"":{""1677156818498"":"" ""}},{""ts"":""2023-02-23T12:53:39.742Z"",""message"":{""1677156819742"":""न""}},{""ts"":""2023-02-23T12:53:41.494Z"",""message"":{""1677156821494"":""व""}},{""ts"":""2023-02-23T12:53:43.095Z"",""message"":{""1677156823095"":""ा""}},{""ts"":""2023-02-23T12:53:44.566Z"",""message"":{""1677156824566"":"" ""}},{""ts"":""2023-02-23T12:53:55.036Z"",""message"":{""1677156835036"":""द""}},{""ts"":""2023-02-23T12:53:57.307Z"",""message"":{""1677156837307"":""bp""}},{""ts"":""2023-02-23T12:53:58.743Z"",""message"":{""1677156838743"":""त""}},{""ts"":""2023-02-23T12:53:59.511Z"",""message"":{""1677156839511"":""ो""}},{""ts"":""2023-02-23T12:54:02.553Z"",""message"":{""1677156842553"":""ठ""}},{""ts"":""2023-02-23T12:54:05.507Z"",""message"":{""1677156845507"":""bp""}},{""ts"":""2023-02-23T12:54:06.778Z"",""message"":{""1677156846778"":""ड""}},{""ts"":""2023-02-23T12:54:09.239Z"",""message"":{""1677156849239"":""ी""}},{""ts"":""2023-02-23T12:54:11.254Z"",""message"":{""1677156851254"":"" ""}},{""ts"":""2023-02-23T12:54:18.801Z"",""message"":{""1677156858801"":""म""}},{""ts"":""2023-02-23T12:54:20.011Z"",""message"":{""1677156860010"":""ु""}},{""ts"":""2023-02-23T12:54:21.109Z"",""message"":{""1677156861109"":""ल""}},{""ts"":""2023-02-23T12:54:22.543Z"",""message"":{""1677156862543"":"" ""}},{""ts"":""2023-02-23T12:54:28.421Z"",""message"":{""1677156868421"":""अ""}},{""ts"":""2023-02-23T12:54:29.095Z"",""message"":{""1677156869095"":""न""}},{""ts"":""2023-02-23T12:54:30.268Z"",""message"":{""1677156870268"":"" ""}},{""ts"":""2023-02-23T12:54:34.167Z"",""message"":{""1677156874167"":""च""}},{""ts"":""2023-02-23T12:54:36.606Z"",""message"":{""1677156876606"":""म""}},{""ts"":""2023-02-23T12:54:38.995Z"",""message"":{""1677156878995"":""क""}},{""ts"":""2023-02-23T12:54:39.828Z"",""message"":{""1677156879828"":"" ""}},{""ts"":""2023-02-23T12:54:56.412Z"",""message"":{""1677156896412"":""ब""}},{""ts"":""2023-02-23T12:54:57.388Z"",""message"":{""1677156897388"":""न""}},{""ts"":""2023-02-23T12:54:58.954Z"",""message"":{""1677156898954"":""े""}},{""ts"":""2023-02-23T12:55:00.237Z"",""message"":{""1677156900237"":"" ""}},{""ts"":""2023-02-23T12:55:03.009Z"",""message"":{""1677156903009"":""म""}},{""ts"":""2023-02-23T12:55:04.135Z"",""message"":{""1677156904135"":""ा""}},{""ts"":""2023-02-23T12:55:06.127Z"",""message"":{""1677156906127"":""म""}},{""ts"":""2023-02-23T12:55:07.755Z"",""message"":{""1677156907755"":"" ""}},{""ts"":""2023-02-23T12:55:12.301Z"",


""message"":{""1677156818506"":{""encoder_invocation_time"":433,
""decoder_invocation_times"":[""587""],
""decoder_invocation_time"":593,
""total_invocation_time"":1027,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.303Z"",""message"":{""1677156824570"":{""encoder_invocation_time"":178,
""decoder_invocation_times"":[""259""],
""decoder_invocation_time"":265,
""total_invocation_time"":443,""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.304Z"",""message"":{""1677156837309"":
{""encoder_invocation_time"":152,
""decoder_invocation_times"":[""214""],
""decoder_invocation_time"":217,
""total_invocation_time"":370,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.306Z"",""message"":{""1677156851259"":
{""encoder_invocation_time"":182,
""decoder_invocation_times"":[""216""],
""decoder_invocation_time"":220,
""total_invocation_time"":402,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.307Z"",""message"":{""1677156862546"":{
""encoder_invocation_time"":276,
""decoder_invocation_times"":[""234""],
""decoder_invocation_time"":238,
""total_invocation_time"":515,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.309Z"",""message"":{""1677156870274"":
{""encoder_invocation_time"":258,
""decoder_invocation_times"":[""222""],
""decoder_invocation_time"":228,
""total_invocation_time"":487,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.311Z"",""message"":{""1677156879831"":
{""encoder_invocation_time"":244,
""decoder_invocation_times"":[""226""],
""decoder_invocation_time"":233,
""total_invocation_time"":477,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.312Z"",""message"":{""1677156900240"":
{""encoder_invocation_time"":247,
""decoder_invocation_times"":[""213""],
""decoder_invocation_time"":217,
""total_invocation_time"":464,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.314Z"",""message"":{""1677156907763"":
{""encoder_invocation_time"":258,
""decoder_invocation_times"":[""215""],
""decoder_invocation_time"":219,
""total_invocation_time"":479,
""text"":""इससे आपका चेहरा मुलायम और चमकदार बना रहेगा।""}}},
{""ts"":""2023-02-23T12:55:12.340Z"",
""message"":{""type"":""o"",""button"":""NEXT"",""time_taken"":193848}},
{""ts"":""2023-02-23T12:55:12.349Z"",""message"":""marking microtask complete""}]"
'''