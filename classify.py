import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    bow = {}
    with open(filepath, 'r') as f:
        for word in f:
            word = word.strip()
            if word in bow:
                bow.update({word: bow.get(word) + 1})
            else:
                if word in vocab:
                    bow.update({word: 1})
                elif None in bow:
                    bow.update({None: bow.get(None) + 1})
                else:
                    bow.update({None: 1})
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    total = 0
    for t in training_data:
        total += 1
    smooth = 1 # smoothing factor
    logprob = {}
    for label in label_list:
        logprob.update({label: 0})
        num = 0
        for t in training_data:
            if t.get('label') == label:
                num += 1
        logprob.update({label: math.log((num + smooth)/(total + 2))})
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    smooth = 1 # smoothing factor
    word_prob = {}
    total = 0
    for w in vocab:
        word_prob.update({w: 0})
    word_prob.update({None: 0})
    for t in training_data:
        if t.get('label') == label: 
            for w in t.get('bow'):
                num = t.get('bow').get(w)
                if w in word_prob:
                    word_prob.update({w: word_prob.get(w) + num})
                else:
                    word_prob.update({None: word_prob.get(None) + num})
                total += num
    for w in word_prob:
        word_prob.update({w: math.log((word_prob.get(w) + smooth*1)/(total + smooth*(len(vocab) + 1)))})
    return word_prob

##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    
    vocab = create_vocabulary(training_directory, cutoff)
    data = load_training_data(vocab, training_directory)
    retval.update({'vocabulary': vocab})
    retval.update({'log prior': prior(data, label_list)})
    retval.update({'log p(w|y=2016)': p_word_given_label(vocab, data, '2016')})
    retval.update({'log p(w|y=2020)': p_word_given_label(vocab, data, '2020')})

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    sum_2016, sum_2020 = model.get('log prior')['2016'], model.get('log prior')['2020']
    b = create_bow(model['vocabulary'], filepath)
    for w in b:
        sum_2016 += model.get('log p(w|y=2016)')[w] * b[w]
        sum_2020 += model.get('log p(w|y=2020)')[w] * b[w]
    
    retval = {}
    retval.update({'log p(y=2020|x)': sum_2020})
    retval.update({'log p(y=2016|x)': sum_2016})
    predict = ''
    if sum_2020 > sum_2016:
        predict = '2020'
    else:
        predict = '2016'
    retval.update({'predicted y': predict})
    return retval
