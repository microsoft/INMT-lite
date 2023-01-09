# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import argparse 


def convert_src_tgt(src_lang, tgt_lang, src_file):
    with open(src_file, 'r', encoding = 'UTF-8') as file, open(f'{src_file}_script_converted.txt', 'w', encoding = 'UTF-8') as out:
        samples = file.read().split('\n')
        for sample in samples: 
            out.write(UnicodeIndicTransliterator.transliterate(sample,src_lang,tgt_lang) + '\n')


def convert_tgt_src(src_lang, tgt_lang, src_file):
    def convert_src_tgt(src_lang, tgt_lang, src_file):
        with open(src_file, 'r', encoding = 'UTF-8') as file,  open(f'{src_file}_script_restored.txt', 'w', encoding = 'UTF-8') as out:
            samples = file.read().split('\n')
            for sample in samples: 
                out.write(UnicodeIndicTransliterator.transliterate(sample,src_lang,tgt_lang) + '\n')


# +
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--src_file", type=str)
    args = parser.parse_args()

    convert_src_tgt(args.src_lang, args.tgt_lang, args.src_file)
    convert_tgt_src(args.src_lang, args.tgt_lang, args.src_file)
  
    
