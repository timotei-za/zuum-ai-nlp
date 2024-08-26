import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from backend.outgen import tokenizer


def preprocess(dataframe):
    print(dataframe.shape)
    for i in range(dataframe.shape[0]):
        try:
            text = dataframe['text'][i].split()
        except:
            print(dataframe['text'][i])
            dataframe = dataframe.drop([i])
            continue
    
    print(dataframe.shape)
    dataframe.to_csv("/Users/priyansha/zuum/venv3-9/src/text_intent.csv", index=False)


# format dataset into utterances | intent
def formatDataset(dataset_dir):
    # flatten dataset so all utterances 
    # are in one column and next column holds associated intent

    dataframe = pd.DataFrame(pd.read_csv(dataset_dir))
    dataframe_f = pd.DataFrame(index=np.arange(dataframe.shape[0]*NUM_INTENTS), columns=['text', 'intent'])
    
    # iterate through every text in dataframe and add to dataframe_f with the column intent next to it
    count = 0
    col_len = 0
    for intent in dataframe.columns:
        col_len = len(dataframe[intent])
        for val in range(col_len):
            text = str(dataframe[intent][val]).replace("\n", "")
            dataframe_f.iloc[val + count*col_len, 0] = text
            dataframe_f.iloc[val + count*col_len, 1] = str(intent)
        count+=1

    return dataframe_f
def find_max_length(utterances, max_len):
    for utternace in utterances:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        try:
            input_ids = tokenizer.encode(utternace, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))

        except: 
            print(utternace)
            continue
        
    print('Max sentence length: ', max_len)

MAX_LENGTH = 30 # max value of a text in the dataset
NUM_INTENTS = 3

if __name__ == "__main__":
    seed_val = 42
    dataset_dir = "/Users/priyansha/zuum/venv3-9/src/three_intents_dataset.csv"
    labels_dict = {'Query_for_Shipment_Location': 0, 'Query_for_Shipment_StartEnd_Date': 1, 'Query_for_Shipments_sent_in_windowed_time_period': 2}

    # script to formatDataset
    #dataframe = formatDataset(dataset_dir)
    #dataframe.to_csv('text_intent.csv', index=False)

    # script to remove all nan in dataset
    dataframe = pd.read_csv("/Users/priyansha/zuum/venv3-9/src/text_intent.csv")
    # preprocess(dataframe)

    # script to change all labels in dataframe into id values
    
    # dataframe_f = pd.DataFrame(index = np.arange(dataframe.shape[0]), columns = ['text', 'intent'])
    # for i in range(dataframe_f.shape[0]):
    #    dataframe_f.loc[i, 'text'] = dataframe['text'][i]
    #    dataframe_f.loc[i, 'intent'] = labels_dict[dataframe['intent'][i]]
    
    # dataframe_f.to_csv("/Users/priyansha/zuum/venv3-9/src/use_text_intent.csv", index=False)

    # give info for dataframe
    # counts, edges, bars = plt.hist(dataframe['intent'])
    # plt.bar_label(bars)
    # plt.show()

    dataframe = pd.DataFrame(pd.read_csv("/Users/priyansha/zuum/venv3-9/src/use_text_intent.csv"))
    
    train, test = train_test_split(dataframe, test_size=0.1, random_state=seed_val)
    train, val = train_test_split(train, test_size=0.1, random_state=seed_val)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    val.to_csv("val.csv", index=False)

    # find max length of sentences in BERT and add special tokens
    # find_max_length(dataframe['text'], 0)
