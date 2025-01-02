# Preprocessing txt file before training

corpus_path = 'D:/Dev/BA/jarlhengstmengel/Data/Wortschatz/deu_news_2016_1M/deu_news_2016_1M-sentences.txt'
mod_corpus = open('D:/Dev/BA/jarlhengstmengel/Preprocessed_Data/run/deu_news_2016_1M-sentences_preprocessed.txt', 'w', encoding='utf8')
for line in open(corpus_path, encoding='utf8'):
    line = line.split('\t')
    mod_corpus.write(line[1])
mod_corpus.close()