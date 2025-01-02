from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.models.callbacks import CallbackAny2Vec
import logging
import os
from datetime import datetime
from somajo import SoMaJo

# setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# define parameters
learning_data = 'Preprocessed_Data/run/deu_news_2016_1M-sentences_preprocessed.txt'
evaluation_data = 'Data/analogies/de_trans_Google_analogies.txt'
model_save_dir = 'E:/Dev/BA/Saved_Models'
eval_score_dir = 'Model_Scores/'
trainable_model = 'E:/Dev/BA/Saved_Models/fastText_2023_06_10_18_45_11.model'

vector_size = 300
window = 5
sg = 0
seed = 42
negative = 10
epochs = 10

# iterator for iterating over corpus
class Corpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        tokenizer = SoMaJo('de_CMC')
        sentence_counter = 0
        with open(self.path, encoding='utf-8') as f:
            sentences = tokenizer.tokenize_text_file(f, paragraph_separator='single_newlines', parallel=1)
            for sentence in sentences:
                if sentence_counter % 5000 == 0:
                    print('Sentences seen in current epoch: {}'.format(sentence_counter))
                sentence_counter += 1
                yield list(token.text for token in sentence)


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print('Starting epoch #{}'.format(self.epoch))

    def on_epoch_end(self, model):
        print('Ending epoch #{}'.format(self.epoch))
        self.epoch += 1


def log_scores(scores, file):
    overal_acc = scores[0]
    
    with open(score_file, 'w') as f:
        for j in score:
            f.write(str(i) + '\n')


if __name__ == '__main__':
    # Start tokenizer
    corpus = Corpus(learning_data)


    # Model initialization
    # model = FastText(vector_size=vector_size, window=window, sg=sg, seed=seed, negative=negative)

    # Or load model
    model = FastText.load(trainable_model)

    # Build or update vocab
    model.build_vocab(corpus_iterable=corpus, update=True)


    # Model training
    model.train(corpus_iterable=corpus, epochs=epochs, total_examples=model.corpus_count,
                total_words=model.corpus_total_words, callbacks=[EpochLogger()])

    print(model)
    model_save = model.wv
    print(model_save)

    date_time_save = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Evaluation
    score = model_save.evaluate_word_analogies(evaluation_data, dummy4unknown=True)
    score_file = os.path.join(eval_score_dir, 'score_' + date_time_save + '.txt')
    with open(score_file, 'w') as f:
        for i in score:
            f.write(str(i) + '\n')

    score = model_save.evaluate_word_analogies(evaluation_data, dummy4unknown=False)
    score_file = os.path.join(eval_score_dir, 'score_' + date_time_save + '_wodummy.txt')
    with open(score_file, 'w') as f:
        for i in score:
            f.write(str(i) + '\n')

    # Saving the model
    print('Saving model...')
    fname = os.path.join(model_save_dir, 'fastText_' + date_time_save + '.model')
    model.save(fname)
    print('Model saved at {}'.format(fname))

