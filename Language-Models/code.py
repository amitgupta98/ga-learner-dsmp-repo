# --------------
# import packages
import re
from collections import Counter
import numpy as np

def edit(word):
    #All edits that are one edit away from `word`.
    
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


# code starts here
# convert the text to lower
def words(text):
    return re.findall(r'\w+', text.lower())

# open the file
WORDS = Counter(words(open(path).read()))


def known(words): 
    return list(w for w in words if w in WORDS)



# --------------
# code starts here
def edits(word):
    
    first_edited_word = []
    second_edited_word = []
    third_edited_word = []
    
    first_edited_word = edit(word)
    # list of all possible words append in the second edit 
    for word in first_edited_word:
        for word2 in edit(word):
            second_edited_word.append(word2)
   # list of all possible words append in the third edit         
    for word in second_edited_word:
        for word2 in edit(word):
            third_edited_word.append(word2)

    return [first_edited_word, second_edited_word, third_edited_word]



# --------------
# code starts here
def probability(word, N=sum(WORDS.values())):
    prob_list = []
    for w in word:
        p_w = WORDS[w] / N
        prob_list.append(p_w)

    return prob_list


# code ends here


# --------------
# code starts here
def correction(word):
    first_edited_word,second_edited_word,third_edited_word = edits(word)
    
    known_word = known([word])
    known_words_1edit = known(first_edited_word)
    known_words_2edit = known(second_edited_word)
    known_words_3edit = known(third_edited_word)
                       
        
    if len(known_word) != 0:
        return word
    
    elif len(known_words_1edit) != 0:
        probability_list = probability(known_words_1edit)
        max_index = np.argmax(probability_list)
        return known_words_1edit[max_index]
    
    elif len(known_words_2edit) != 0:
        probability_list = probability(known_words_2edit)
        max_index = np.argmax(probability_list)
        return known_words_2edit[max_index]
    
    elif len(known_words_3edit) != 0:
        probability_list = probability(known_words_3edit)
        max_index = np.argmax(probability_list)
        return known_words_3edit[max_index]
    
    else:
        return word
    
    


# code ends here


# --------------
""" After filling and submitting the feedback form, click the Submit button of the codeblock"""


