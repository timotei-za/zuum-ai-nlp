import gensim
import transformers
import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word.context_word_embs as nawcwe
import nlpaug.flow as nafc
from nlpaug.util import Action
# downloading models to current directory
from nlpaug.util.file.download import DownloadUtil

from colorama import Fore
from transformers import BertModel, BertTokenizer
import pandas
import numpy as np


def print_and_highlight_diff(orig_text, new_texts):
    """ A simple diff viewer for augmented texts. """
    orig_split = orig_text.split()
    print(f"Original: {len(orig_split)}\n{orig_text}\n")
    for new_text in new_texts:
        print(f"Augmented: {len(new_text.split())}")
        for i, word in enumerate(new_text.split()):
            if i < len(orig_split) and word == orig_split[i]:
                print(Fore.RESET + word, end=" ")
            else:
                print(Fore.RED + word + Fore.RESET, end=" ")
        print()


if __name__ == "__main__":
    # downloading word2vec embeddings
    # DownloadUtil.download_word2vec(dest_dir='.')
    # model_path = '/Users/priyansha/zuum/venv3-9/src/GoogleNews-vectors-negative300.bin'
    # downloading fasttext embeddings
    # DownloadUtil.download_fasttext(dest_dir='.',model_name='crawl-300d-2M')
    # downloading glove embeddings
    # DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.',)

    model = BertModel.from_pretrained("/Users/priyansha/zuum/venv3-9/src/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("/Users/priyansha/zuum/venv3-9/src/bert-base-uncased")

    dataset = pandas.read_csv("/Users/priyansha/zuum/venv3-9/src/intents_dataset.csv", header=0)
    dataframe = pandas.DataFrame(dataset)

    aug = nawcwe.ContextualWordEmbsAug(model_path='bert-base-uncased', aug_max=5, aug_p=0.3,
                                       stopwords=["shipment", "truck"], stopwords_regex="[0-9]+ ")

    append_df = pandas.DataFrame(index=np.arange(4 * dataframe.shape[0]), columns=np.arange(dataframe.shape[1]))

    count = 0
    for column in dataframe.columns:
        if (count == 3):
            break
        for val in range(len(dataframe[column])):
            text = dataframe[column][val]
            augmented_text = aug.augment(text, n=4)
            for t in range(len(augmented_text)):
                append_df.iloc[val + 100 * t, dataframe.columns.get_loc(column)] = augmented_text[t].replace(" ' ", "'")
        count += 1

    append_df.to_csv('out.csv', index=False)
