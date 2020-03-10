#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 13:22
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : numpy_balance.py

# refer to balance.py
import numpy as np
import tqdm
'''
num_classes = 10
step 1 -- r = r*c               [1000*10] * [10] 
step 2 -- d = r.count()         [10]
step 3 -- score = sum(min(d,1)) float
step 4 -- max(score) by c
'''

def _get_predicts(predicts, coefficients):
    # return np.einsum("ij,j->ij", (predicts, coefficients))
    return predicts * coefficients

def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = np.argmax(predicts, axis=-1)
    counter = np.bincount(labels, minlength=predicts.shape[1])
    return counter

def _compute_score_with_coefficients(predicts, coefficients):
    counter = _get_labels_distribution(predicts, coefficients)
    counter = counter * 100 / len(predicts)
    max_scores = np.ones(len(coefficients),dtype=np.float) * 100 / len(coefficients)

    result = np.min(np.stack((counter, max_scores), axis=0), axis=0)

    return np.sum(result)

def _find_best_coefficients(predicts, coefficients, alpha=0.001, iterations=100):
    best_coefficients = np.copy(coefficients)
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in tqdm.trange(iterations):
        counter = _get_labels_distribution(predicts, coefficients)
        label = int(np.argmax(counter))
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)
        if score > best_score:
            best_score = score
            best_coefficients = np.copy(coefficients)

    return best_coefficients

def get_balance_label(y,start_alpha=0.01,min_alpha=0.0001,):
    coefs = np.ones(y.shape[1])
    last_score = _compute_score_with_coefficients(y, coefs)
    alpha = start_alpha
    print("Start score", last_score)

    while alpha >= min_alpha:
        coefs = _find_best_coefficients(y, coefs, iterations=3000, alpha=alpha)
        new_score = _compute_score_with_coefficients(y, coefs)

        if new_score <= last_score:
            alpha *= 0.5

        last_score = new_score
        print("Score: {}, alpha: {}".format(last_score, alpha))

    predicts = _get_predicts(y, coefs)
    return predicts

if __name__ == '__main__':
    np.random.seed(1)
    y = np.random.randn(40, 40)
    r = get_balance_label(y)
    o = np.argmax(r, axis=1)
    counter = np.bincount(o, minlength=r.shape[1])
    print(r,r.shape,counter)