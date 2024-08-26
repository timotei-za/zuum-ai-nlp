from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import matplotlib as plt

if __name__ == "__main__":

    test_text = pd.read_csv("test.csv")
    text = test_text['text'][0]

    tokenizer = AutoTokenizer.from_pretrained("/Users/priyansha/zuum/venv3-9/src/n_distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("/Users/priyansha/zuum/venv3-9/src/n_distilbert-base-uncased")
    
    accuracy = 0
    i = 0

    stats = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    text_missed = []

    for text in test_text['text']:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
    
        predicted_class_id = logits.argmax().item()
        if predicted_class_id == test_text['labels'][i]:
            accuracy += 1
            stats[predicted_class_id][0]+=1
        else:
            text_missed.append([text, (predicted_class_id, test_text['labels'][i])])
        stats[test_text['labels'][i]][1] += 1
        i+=1
    
    print("*"*10, '\nACCURACY:', accuracy/i)
    print("*"*10)

    print(stats)
    print(text_missed)

