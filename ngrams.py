import os, shutil, random
import numpy as np

ifname = 'aspect.csv'

perf = 'совершенный'
impf = 'несовершенный'
labl = 'двувидовой'

n_lo = 2 # minimum n for n-grams
n_hi = 4 # maximum n for n-grams

root = os.path.join('.', 'aspects')
pdir = 'perfect'
idir = 'imperfect'

aspects = [perf, impf]
dirs = [pdir, idir]

def mkdir(dirname, remake=True):
    """
    Safe makedir.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif remake:
        print('Folder {} exists -- remaking'.format(dirname))
        shutil.rmtree(dirname)    
        os.makedirs(dirname)

def word_n_grams(word, n):
    """
    Get n-grams of length n from word.
    """
    return [word[i:i+n] for i in range(len(word)-n+1)]

def word_n_grams_r(word, n, max_n):
    """
    Recursively get all n-grams of lengths from n up to max_n.
    """
    if n > max_n:
        return []
    new_grams = word_n_grams(word, n)
    if new_grams == []:
        return []
    return new_grams + word_n_grams_r(word, n+1, max_n)

def get_lists(ifname=ifname):
    """
    Read the data.
    """
    asp_dict = {}
    with open(ifname, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            parts = line.strip('\n').split(',')
            if len(parts) == 2:
                lemma, aspect = parts[0], parts[1]
                asp_dict[aspect] = asp_dict.get(aspect, []) + [lemma]
    for asp in asp_dict:
        asp_dict[asp] = list(set(asp_dict[asp]))
        asp_dict[asp].sort()
    return asp_dict

def output_aspect(asp, verb_list, root):
    """
    Output the data for one aspect.
    """
    mkdir(root)
    for verb in verb_list:
        ofile = open('{}{}.txt'.format(root, verb), 'w', encoding='utf-8')
        ofile.write(' '.join(word_n_grams_r(verb, n_lo, n_hi)))
        ofile.close()

def output_aspects(asp_dict):
    """
    Output all data sorted into folders.
    """
    max_len = min(len(asp_dict[perf]), len(asp_dict[impf]))
    for aspect, adir in zip(aspects, dirs):
        output_aspect(aspect, random.sample(asp_dict[aspect], max_len), os.path.join(root, adir))

def make_n_grams(verb_list):
    """
    Make n-grams for each verb in verb_list.
    """
    n_grammized_list = []
    for verb in verb_list:
        n_grammized_list.append(' '.join(word_n_grams_r(verb, n_lo, n_hi)))
    return n_grammized_list

def aspect_dataset(ifname=ifname):
    """
    Produce aspect dataset.
    """
    asp_dict = get_lists(ifname)
    perf_verbs = asp_dict[perf]
    impf_verbs = asp_dict[impf]
    max_len = min(len(perf_verbs), len(impf_verbs))
    perf_verbs = perf_verbs[:max_len]
    impf_verbs = impf_verbs[:max_len]
    perf_list = make_n_grams(perf_verbs)
    impf_list = make_n_grams(impf_verbs)
    verbs = np.asarray(perf_verbs + impf_verbs)
    X = np.asarray(perf_list + impf_list)
    y = np.asarray([0] * max_len + [1] * max_len)
    return verbs, X, y, ['perfect', 'imperfect']

def load_test_data(fname):
    """
    Load test data.
    """
    with open(fname, 'r', encoding='utf-8') as ifile:
        verbs = [line.strip() for line in ifile if line != '\n']
    X = np.asarray(make_n_grams(verbs))
    y = np.asarray([0] * len(verbs))
    return np.asarray(verbs), X, y

if __name__ == '__main__':
    asp_dict = get_lists()
    output_aspects(asp_dict)

