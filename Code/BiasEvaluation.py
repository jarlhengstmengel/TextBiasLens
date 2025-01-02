from gensim.models import fasttext
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Script for evaluating Bias

# define parameters
model_path = 'D:/Dev/BA/jarlhengstmengel/Saved_Models/cc.de.300.bin/cc.de.300.bin'
defining_sets_path = 'D:/Dev/BA/jarlhengstmengel/Analogies/DefiningSets/jusctConcept'
comparing_set_txt = 'D:/Dev/BA/jarlhengstmengel/Analogies/School_Level.txt'
save_dir = 'D:/Dev/BA/jarlhengstmengel/Bias_Analyses'

dom_group = 'Weiß'
bias_strictness = 1


def read_from_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        input = f.read()
        input_arr = input.split(', ')
        return input_arr


def calc_pair_to_center_dif(wv_1, wv_2):
    # calculate the center of a vector pair and their differences to that center
    center = np.mean([wv_1, wv_2], axis=0)
    wv_1_center_dif = wv_1 - center
    wv_2_center_dif = wv_2 - center
    return wv_1_center_dif, wv_2_center_dif


def get_bias_subspace(def_pairs, n_components):
    i = 1

    # calculate differences of bias defining pairs with their center and combine the resulting vectors to one matrix
    wv_1_center_dif, wv_2_center_dif = calc_pair_to_center_dif(wv_1=def_pairs[0][0],
                                                               wv_2=def_pairs[0][1])
    center_diff_arr = np.array([wv_1_center_dif])
    center_diff_arr = np.append(center_diff_arr, [wv_2_center_dif], axis=0)
    while i in range(np.size(def_set, 0)):
        wv_1_center_dif, wv_2_center_dif = calc_pair_to_center_dif(wv_1=def_pairs[i][0],
                                                                   wv_2=def_pairs[i][1])
        center_diff_arr = np.append(center_diff_arr, [wv_1_center_dif], axis=0)
        center_diff_arr = np.append(center_diff_arr, [wv_2_center_dif], axis=0)
        i = i + 1

    # analyze principal components
    pca = PCA(n_components=n_components)
    pca.fit(center_diff_arr)
    first_pc = pca.components_[0]
    test_components = pca.components_

    # return bias_subspace
    return first_pc, pca.explained_variance_ratio_


def calc_direct_bias(comparing_word_vectors, bias_subspace, c):
    db = 0
    for w in comparing_word_vectors:
        db = db + (np.abs(np.dot(w, bias_subspace) / np.dot(np.linalg.norm(w), np.linalg.norm(bias_subspace))) ** c)
    db = db / np.size(comparing_word_vectors, axis=0)
    return db


def get_array_normed_vectors(word_vectors, words):
    i = 1
    normed_word_vectors_array = np.array([word_vectors.get_vector(words[0], norm=True)])
    while i in range(np.size(words)):
        normed_word_vectors_array = np.append(normed_word_vectors_array, [word_vectors.get_vector(words[i], norm=True)], axis=0)
        i = i + 1
    return normed_word_vectors_array


if __name__ == '__main__':
    # load model
    model = fasttext.load_facebook_model(model_path)
    model_wv = model.wv

    for def_set_file in os.listdir(defining_sets_path):
        def_set_txt = os.path.join(defining_sets_path, def_set_file)
        if os.path.isfile(def_set_txt):
            def_set = read_from_txt(def_set_txt)
            def_set_normed_vectors = get_array_normed_vectors(model_wv, def_set)
            i = 1
            def_set_arr = np.array([[model_wv.get_vector(dom_group, norm=True), def_set_normed_vectors[0]]])
            while i in range(np.size(def_set, axis=0)):
                def_set_arr = np.append(def_set_arr, [[model_wv.get_vector(dom_group, norm=True), def_set_normed_vectors[i]]], axis=0)
                i = i + 1

            # calculate bias subspace with PCA
            bias_subspace, pca_variance_ratio = get_bias_subspace(def_pairs=def_set_arr, n_components=5)

            date_time_save = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

            # plot principal components
            plt.bar(('PC1', 'PC2', 'PC3', 'PC4', 'PC5'), pca_variance_ratio)
            plt.savefig(os.path.join(save_dir, 'PCA_' + date_time_save + '.png'))

            comparing_set = read_from_txt(comparing_set_txt)
            comparing_set_normed_vectors = get_array_normed_vectors(model_wv, comparing_set)
            db = calc_direct_bias(comparing_set_normed_vectors, bias_subspace, bias_strictness)
            print(db)

            with open(os.path.join(save_dir, 'bias_analyses_' + date_time_save + '.txt'), 'w', encoding='utf-8') as f:
                f.write('Used Model: ' + model_path + '\n')
                f.write('Used defining set: ' + str(def_set) + '\n')
                f.write('Used comparing set: ' + str(comparing_set) + '\n')
                f.write('PCA variance ratio (PC1, PC2, PC3, PC4, PC5): ' + str(pca_variance_ratio) + '\n')
                f.write('Bias score comparing set to bias subspace: ' + str(db) + '\n')
                f.write('Bias strictness parameter: ' + str(bias_strictness) + '\n')

