from gensim.models import fasttext
import logging
import numpy as np
from datetime import datetime
import os

# script for generating analogies
# setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# define parameters
model_path = 'Saved_Models/cc.de.300.bin/cc.de.300.bin'
save_dir = 'D:/Dev/BA/jarlhengstmengel/Analogies'
defining_sets_path = 'D:/Dev/BA/jarlhengstmengel/Analogies/DefiningSets/justConcept'
comparing_set = 'D:/Dev/BA/jarlhengstmengel/Analogies/School_Level.txt'
threshold = 0.9

# attention with the file order in folder
dom_groups = ['Weiß']


def get_best_analogie(word_vectors, word_a, word_b, word_x, threshold=1.0):
    # a is to x as b is to y
    a_vec = word_vectors.get_vector(word_a, norm=True)
    b_vec = word_vectors.get_vector(word_b, norm=True)
    x_vec = word_vectors.get_vector(word_x, norm=True)

    mod_vec = word_vectors.get_normed_vectors()
    highest_score = 0
    highest_score_vec = np.zeros(a_vec.shape)

    # find best cosine similarity score
    for y_vec in mod_vec:
        if np.linalg.norm(x_vec - y_vec, ord=2) <= threshold:
            score = np.dot(a_vec - b_vec, x_vec - y_vec)
            if score > highest_score:
                highest_score = score
                highest_score_vec = y_vec
    return highest_score_vec, highest_score


def generating_analogies(word_vectos, def_terms, an_terms, dom_term, threshold=1.0):
    analogies = np.array([['a', 'x', 'b', 'y']])
    for def_word in def_terms:
        for an_word in an_terms:
            y_1_vec, y_1_vec_score = get_best_analogie(word_vectors=word_vectos, word_a=def_word, word_b=dom_term,
                                                       word_x=an_word, threshold=threshold)
            y_2_vec, y_2_vec_score = get_best_analogie(word_vectors=word_vectos, word_a=dom_term, word_b=def_word,
                                                       word_x=an_word, threshold=threshold)
            y_1 = word_vectos.most_similar(positive=y_1_vec, topn=1)
            y_2 = word_vectos.most_similar(positive=y_2_vec, topn=1)

            y_1 = np.array(y_1)
            y_2 = np.array(y_2)

            analogies = np.append(analogies, [[def_word, an_word, dom_term, y_1[0][0]]], axis=0)
            analogies = np.append(analogies, [[dom_term, an_word, def_word, y_2[0][0]]], axis=0)
    analogies = np.delete(analogies, 0, 0)
    return analogies


def read_from_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        input = f.read()
        input_arr = input.split(', ')
        return input_arr


def write_analogies_to_txt(path, analogies, setting, threshold):
    date_time_save = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    j = 0
    with open(os.path.join(path, 'analogies_' + date_time_save + '_' + setting + '.txt'), 'w', encoding='utf-8') \
            as file:
        for element in np.nditer(analogies):
            if j == 0 or not(j % 4 == 0):
                file.write(str(element) + '\t')
            if j % 4 == 0 and not j == 0:
                file.write('\n' + str(element) + '\t')
            j = j + 1
        file.write('\n' + 'Defining Set: ' + setting)
        file.write('\n' + 'Threshold: ' + str(threshold))


if __name__ == '__main__':
    # load model
    model = fasttext.load_facebook_model(model_path)
    model_wv = model.wv
    i = 0

    # load comparing set
    if os.path.isfile(comparing_set):
        comparing_set_txt = read_from_txt(comparing_set)

        # load defining sets and generate analogies
        for def_set_file in os.listdir(defining_sets_path):
            def_set_txt = os.path.join(defining_sets_path, def_set_file)
            if os.path.isfile(def_set_txt):
                def_set_arr = read_from_txt(def_set_txt)
                generated_analogies = generating_analogies(word_vectos=model_wv, def_terms=def_set_arr,
                                                           an_terms=comparing_set_txt, dom_term=dom_groups[i], threshold=threshold)
                # generated_analogies = np.array([['a', 'x', 'b', 'y']])
                write_analogies_to_txt(path=save_dir, analogies=generated_analogies, setting=def_set_file,
                                       threshold=threshold)
            else:
                print('Defining set/file does not exist')
            i = i + 1
    else:
        print('Comparing set/file does not exist')
