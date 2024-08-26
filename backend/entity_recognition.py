import spacy

def entity_recognizer(user_input):
    nlp = spacy.load("Custom_NER/trained_models/output_status(best-0.7)/model-best")
    text = user_input
    print(f"User Input: {text}")

    doc = nlp(text)
    id = None  # Initialize id with None

    for ent in doc.ents:
        # Print the recognized text and its corresponding label
        print(f"Entity: {ent.text}, Label: {ent.label_}")
        if ent.label_ == "SHIPMENT ID":
            try:
                id = int(ent.text)
                break  # Exit the loop once the shipment ID is found and assigned
            except ValueError:
                continue

    if id is None:
        raise ValueError("SHIPMENT ID not found in the input")

    return id

if __name__ == '__main__':
    print(entity_recognizer('what is the status of shipment id 46'))

