import nltk         #大資料夾中有許多小資料夾
import numpy as np
nltk.download('punkt')  #斷詞工具
from nltk.stem.porter import PorterStemmer  #詞幹提取

stemmer = PorterStemmer()  #命名

def tokenize(sentence):     
    return nltk.word_tokenize(sentence)         

def stem(word):     
    return stemmer.stem(word.lower())      #word目標字；lower小寫         

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)    #建立矩陣且設定類型
    
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0  #若有出現某字，則出現頻率令為1 (簡化)
            
    return bag