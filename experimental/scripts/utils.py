import json
import pandas as pd    
import logging

def read_jsonl(jsonl_path):
    jsonObj = pd.read_json(path_or_buf=jsonl_path, lines=True)
    source_sentences = jsonObj['premise']
    applicable = []
    for sen in source_sentences:
        if len(sen.split(' ')) > 5: 
            applicable.append(sen) 
    logging.debug(f'{len(applicable)} are being used.')
    return applicable 

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
    
        


