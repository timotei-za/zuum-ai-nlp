import spacy


def entity_recognizer(user_input):
    # print(user_input)
    nlp = spacy.load("backend/model-best")
    text = user_input
    # print(text)

    doc = nlp(text)

    for ent in doc.ents:
        # Print the recognized text and its corresponding label
        ###
        print(f'\n*** Model recognized the following entities: ***')
        print(f'*** ENT-TEXT -> {ent.text} <- ENT-LABEL -> {ent.label_} ***\n')
        ###
        if ent.label_ == "SHIPMENT ID":
            try:
                id = int(ent.text) 
            except:
                continue
    return id


if __name__ == '__main__':
    entity_recognizer('what is the status of shipment id 46')
