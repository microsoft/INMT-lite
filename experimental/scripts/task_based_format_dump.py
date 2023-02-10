import json 
import logging
import argparse
from utils import read_json, dump_task_wise_json

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file_path", type = str)
    parser.add_argument("--nos", type = int, default = 40)
    args = parser.parse_args()
    source_sentences, translations, BOW = read_json(args.dump_file_path)
    logging.debug('All Tasks Sentences Dump loaded.')
    task_types = ['baseline', 'dynamic-bow', 'next-word-BOW', 'next-word-dropdown', 'post-edited', 'static-BOW']
    task_specific_counter, number_of_samples_per_task = 0, 36
    for task_type in task_types: 
        dump_task_wise_json(source_sentences[task_specific_counter :  task_specific_counter + number_of_samples_per_task], translations[task_specific_counter :  task_specific_counter + number_of_samples_per_task], args.dump_file_path, task_type)
        task_specific_counter += number_of_samples_per_task
        logging.debug(f'Finished for {task_type}')



