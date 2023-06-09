import json
import pandas as pd    
import logging


def read_json(json_path):
    jsonObj = pd.read_json(path_or_buf=json_path, lines=True)
    source_sentences = jsonObj['sentence']
    translations = jsonObj["providedTranslation"]
    BOW = jsonObj["BOW"]
    return source_sentences, translations, BOW

def read_jsonl(jsonl_path):
    jsonObj = pd.read_json(path_or_buf=jsonl_path, lines=True)
    source_sentences = jsonObj['premise']
    applicable = []
    for sen in source_sentences:
        if len(sen.split(' ')) > 5: 
            applicable.append(sen) 
    logging.debug(f'{len(applicable)} are being used.')
    return applicable 

def dump_task_wise_json(src_samples, predictions, dump_file_name, task_type): 
    obj_cluster = []
    for src, pred in zip(src_samples, predictions):
        if task_type in ['baseline', 'dynamic-bow', 'next-word-BOW', 'next-word-dropdown']: 
            formatted_obj = {'sentence': src}
            obj_cluster.append(formatted_obj)
        if task_type == 'post-edited': 
            formatted_obj = {'sentence': src, 'providedTranslation': pred}
            obj_cluster.append(formatted_obj)
        if task_type == 'static-BOW': 
            bow = pred.split(' ')
            formatted_obj = {'sentence': src,'providedTranslation': pred, 'BOW': bow}
            obj_cluster.append(formatted_obj)
    
    final_file_name =  task_type + '_' + dump_file_name 
    with open(final_file_name, "w") as outfile:
        for item in obj_cluster:
            outfile.write(json.dumps(item, ensure_ascii = False) + '\n')
        return True
    

def dump_json(src_samples, predictions, dump_file_name): 
    obj_cluster = []
    for src, pred in zip(src_samples, predictions):
        bow = pred.split(' ')
        formatted_obj = {'sentence': src,'providedTranslation': pred, 'BOW': bow}
        obj_cluster.append(formatted_obj)
     
    with open(dump_file_name, "w") as outfile:
        for item in obj_cluster:
            outfile.write(json.dumps(item, ensure_ascii = False) + '\n')
        return True
    
def get_interface_mapping(task_name): 
    if 'Next Word BOW' in task_name: 
        return 'NWBOW'
    if 'Next Word Dropdown' in task_name: 
        return 'NWD' 
    if 'BASELINE' in task_name: 
        return 'B'
    if 'DYNAMIC_BOW' in task_name:
        return 'DBOW' 
    if 'STATIC_BOW' in task_name: 
        return 'SBOW' 
    if 'POST_EDITED' in task_name: 
        return 'PE'
    return 'Invalid Interface Name.'


