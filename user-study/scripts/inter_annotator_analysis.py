import json 
from utils import get_interface_mapping
from sklearn.metrics import cohen_kappa_score, f1_score 
import numpy as np
analysis_path = '/home/t-hdiddee/INMT-lite/user-study/data/validation_score.dsv'

with open(analysis_path,'r') as file: 
    records = file.read().strip().split('\n')
print(f'{len(records)} are the number of records being analysed.')

def unpaired_compute_cohens_cappa(rater1, rater2):
    # k = cohen_kappa_score(rater1, rater2)
    f1 = f1_score(rater1, rater2, average = "weighted")
    return f1

def compute_cohens_cappa(paired_score_for_interface):

    rater1, rater2 = [], []
    for ele in paired_score_for_interface:
        try:
            rater1.append(ele[0])
            rater2.append(ele[1])
        except: 
            print(ele)
    k = cohen_kappa_score(rater1, rater2)
    f1 = f1_score(rater1, rater2, average = "weighted")
    return k, f1
# NORMALIZE THE SCORES - In order to stabilize the range across the inter annotator agreement was being calculated. 

def normalize_score_per_instruction(raw_score):
    if raw_score < 10: 
        return 10 
    elif raw_score < 29:
        return 25
    elif raw_score < 50: 
        return 35
    elif raw_score < 69: 
        return 60 
    elif raw_score < 90: 
        return 80
    return 90


def print_and_dump(instance):
    print(instance)
    with open('./results/interannotator_analysis.txt', 'a') as file:
        file.write(instance + '\n')
    
# interfacewise_clusters = {} # interface: scores of all sentences (multiple) 
# sentencewise_clusters = {} # sentence - scores of all formats

# for record in records:
#     try: 
#         tid, scorer_id, sid, source, translation, score, mode, _ = record.split('$')
#         interface = get_interface_mapping(mode)
#         if sid in sentencewise_clusters:
#             sentencewise_clusters[sid].append((interface, int(score)))
#         else: 
#             sentencewise_clusters[sid] = [(interface,int(score))]
#         if interface in interfacewise_clusters:
#             interfacewise_clusters[interface].append(int(score))
#         else: 
#             interfacewise_clusters[interface] = [int(score)]   
#     except: 
#         print(record)
     

# for interface in interfacewise_clusters:
#     try:
#         print_and_dump(f'Average sentence quality for {interface} is {np.average(interfacewise_clusters[interface])}')
#     except: 
#         print(interface)
        

# Generating sentence wise - interface wise mapping to compute the average IAA per interface

# DBOW3, SBOW3, NWD3, NWBOW3, PE3, B3 = [],[],[],[],[],[]
# DBOW2, SBOW2, NWD2, NWBOW2, PE2, B2 = [],[],[],[],[],[]
# DBOW, SBOW, NWD, NWBOW, PE, B = [],[],[],[],[],[]
# print(len(sentencewise_clusters))
# negated = 0
# for sid in sentencewise_clusters:
#     if len(sentencewise_clusters[sid]) == 21 or len(sentencewise_clusters[sid]) == 27 or len(sentencewise_clusters[sid]) == 24:
#         dbow, sbow, nwd, nwbow, pe, b = [],[],[],[],[],[]
#         for mappings in sentencewise_clusters[sid]:
#             if mappings[0] == 'NWD':
#                 nwd.append(normalize_score_per_instruction(mappings[1]))
#             if mappings[0] == 'NWBOW':
#                 nwbow.append(normalize_score_per_instruction(mappings[1]))
#             if mappings[0] == 'DBOW':
#                 dbow.append(normalize_score_per_instruction(mappings[1]))
#             if mappings[0] == 'SBOW':
#                 sbow.append(normalize_score_per_instruction(mappings[1]))
#             if mappings[0] == 'B':
#                 b.append(normalize_score_per_instruction(mappings[1]))
#             if mappings[0] == 'PE':
#                 pe.append(normalize_score_per_instruction(mappings[1]))
            

#         dbow.sort(reverse = True)
#         nwbow.sort(reverse = True)
#         sbow.sort(reverse = True)
#         b.sort(reverse = True)
#         pe.sort(reverse = True)
#         nwd.sort(reverse = True)

#         DBOW.append(dbow[:2])
#         NWBOW.append(nwbow[:2])
#         SBOW.append(sbow[:2])
#         B.append(b[:2])
#         PE.append(pe[:2])
#         NWD.append(nwd[:2])


#         DBOW2.append(dbow[1:3])
#         NWBOW2.append(nwbow[1:3])
#         SBOW2.append(sbow[1:3])
#         B2.append(b[1:3])
#         PE2.append(pe[1:3])
#         NWD2.append(nwd[1:3])

#         DBOW3.append(dbow[::2])
#         NWBOW3.append(nwbow[::2])
#         SBOW3.append(sbow[::2])
#         B3.append(b[::2])
#         PE3.append(pe[::2])
#         NWD3.append(nwd[::2])
#     else: 
#         negated += 1    
# print(f'{negated} are negated samples.')

# # Compute Pair-Wise Cohen's Kappa 
# interface_score_pairs = [B, B2, B3, PE, PE2, PE3, SBOW, SBOW2, SBOW3, DBOW, DBOW2, DBOW3, NWBOW, NWBOW2, NWBOW3, NWD, NWD2, NWD3]
# interface_identifiers = ['B','B','B','PE','PE','PE','SBOW','SBOW','SBOW','DBOW','DBOW','DBOW','NWBOW','NWBOW','NWBOW','NWD','NWD','NWD']
# for idx, interface in enumerate(interface_score_pairs):
#     iaa, f1 = compute_cohens_cappa(interface)
#     print_and_dump(f'For interface {interface_identifiers[idx]} the pair wise inter-annotator agreement is {iaa} and F1-Score is {f1}.')
#     idx +=1 
    

# B_STD, PE_STD, SBOW_STD, DBOW_STD, NWD_STD, NWBOW_STD = [],[],[],[],[],[]
# for sentence_stats in zip(B, PE, SBOW, DBOW, NWD, NWBOW):
#     B_STD.append(np.nanstd(sentence_stats[0]))
#     PE_STD.append(np.nanstd(sentence_stats[1]))
#     SBOW_STD.append(np.nanstd(sentence_stats[2]))
#     DBOW_STD.append(np.nanstd(sentence_stats[3]))
#     NWD_STD.append(np.nanstd(sentence_stats[4]))
#     NWBOW_STD.append(np.nanstd(sentence_stats[5]))
    
# interface_std = [B_STD, PE_STD, SBOW_STD, DBOW_STD, NWD_STD, NWBOW_STD]
# interface_identifiers = ['B','PE','SBOW','DBOW','NWD','NWBOW']
# idx = 0 
# for interface in interface_std: 
#     obj = {interface_identifiers[idx]: ('Avg STD',np.nanmean(interface))}
#     idx += 1
#     with open('./results/analysis_stats.json', 'a') as f:
#         f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        

# for interface in interfacewise_clusters:
#     try:
#         print_and_dump(f'Average sentence quality for {interface} is {np.average(interfacewise_clusters[interface])}')
#         obj = {interface: ('Avg SQ',np.average(interfacewise_clusters[interface]))}
#         with open('./results/analysis_stats.json', 'a') as f:
#             f.write(json.dumps(obj, ensure_ascii=False) + '\n')
#     except: 
#         print(interface)


# Computing Avg scores for all the sentences per interface. 

sentencewise_clusters = {} # sentence - scores of all formats

for record in records:
    try: 
        tid, scorer_id, sid, source, translation, score, mode, _ = record.split('$')
        interface = get_interface_mapping(mode)
        if sid in sentencewise_clusters:
            sentencewise_clusters[sid].append((interface, normalize_score_per_instruction(int(score))))
        else: 
            sentencewise_clusters[sid] = [(interface, normalize_score_per_instruction(int(score)))]
    except: 
        print(record)


names = ['B', 'PE', 'SBOW', 'DBOW', 'NWBOW', 'NWD']
B, PE, SBOW, DBOW, NWBOW, NWD = [],[],[],[],[],[]

analyzed_samples = 0
for sentence in sentencewise_clusters: 
    b, pe, sbow, dbow, nwbow, nwd = [],[],[],[],[],[]
    for scores in sentencewise_clusters[sentence]:
        if scores[0] == 'B':
            b.append(scores[1]) 
        if scores[0] == 'PE': 
            pe.append(scores[1])
        if scores[0] == 'SBOW': 
            sbow.append(scores[1])
        if scores[0] == 'DBOW': 
            dbow.append(scores[1])
        if scores[0] == 'NWBOW': 
            nwbow.append(scores[1])
        if scores[0] == 'NWD': 
            nwd.append(scores[1])
        analyzed_samples += 1
    if len(b) != 0 and len(pe) != 0 and len(sbow) != 0 and len(nwbow) != 0 and len(dbow)!= 0 and len(nwd) != 0:  
        B.append(normalize_score_per_instruction(np.average(b)))
        PE.append(normalize_score_per_instruction(np.average(pe)))
        SBOW.append(normalize_score_per_instruction(np.average(sbow)))
        DBOW.append(normalize_score_per_instruction(np.average(dbow)))
        NWBOW.append(normalize_score_per_instruction(np.average(nwbow)))
        NWD.append(normalize_score_per_instruction(np.average(nwd)))
        
        # B.append(max(b))
        # PE.append(max(pe))
        # SBOW.append(max(sbow))
        # DBOW.append(max(dbow))
        # NWBOW.append(max(nwbow))
        # NWD.append(max(nwd))
        
        # B.append(int(min(b)))
        # PE.append(int(min(pe)))
        # SBOW.append(int(min(sbow)))
        # DBOW.append(int(min(dbow)))
        # NWBOW.append(int(min(nwbow)))
        # NWD.append(int(min(nwd)))
 
 

print(len(B), len(PE), len(SBOW), len(DBOW), len(NWBOW), len(NWD))
print(B[:10])
print(PE[:10])
pairwise_markers = [B, PE, SBOW, DBOW, NWBOW, NWD]
pairwise_interface = []
for interface_idx in range(len(pairwise_markers)):
    interface = []
    for pair_interface_idx in range(len(pairwise_markers)):
            interface.append(unpaired_compute_cohens_cappa(pairwise_markers[interface_idx],pairwise_markers[pair_interface_idx])) 
    pairwise_interface.append(interface)
    
print(pairwise_interface)
    
                          
    
    
    
    
    