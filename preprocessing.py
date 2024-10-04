from nltk.tokenize import word_tokenize
from collections import Counter
from collections import Counter


def preprocess_captions(annotations, min_word_freq=5):
    all_captions = [ann['caption'] for ann in annotations]
    tokens = [word_tokenize(caption.lower()) for caption in all_captions]
    
    flat_list = [item for sublist in tokens for item in sublist]
    
    counter = Counter(flat_list)
    vocab = {word: idx + 4 for idx, (word, count) in enumerate(counter.items()) if count >= min_word_freq}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    
    return vocab



