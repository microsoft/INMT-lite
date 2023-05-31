# from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
# from indicnlp import common
# from indicnlp import loader

# INDIC_NLP_RESOURCES= '/home/t-hdiddee/inmt/indic_nlp_library/indicnlp/' # Add path to local package 
# common.set_resources_path(INDIC_NLP_RESOURCES)
# loader.load()


# sent = 'PUSHPA Isliye padha likha rahein hain tere ko ? Isliye ?'
# trans = UnicodeIndicTransliterator.transliterate(sent, "en", "hi")
# # trans = ItransTransliterator.from_itrans(sent, "hi")
# print(trans)
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# the text to be transliterated
text = "Apa sabhii kaa yahaan svaagat hai."
  
# printing the transliterated text
print(transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI))