# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
# from transformers import BertModel, BertTokenizer, T5ForConditionalGeneration, BertLMHeadModel

# can generate text with Bert, but its pretty bad
# the T5 i found acts as a spell checker

# bert_url = './zuum_env/bert-base-uncased'
# global_url = 'bert-base-uncased'
# t5_url = 't5-small'

# model = T5ForConditionalGeneration.from_pretrained(t5_url)
# tokenizer = AutoTokenizer.from_pretrained(t5_url)

# from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("lucyknada/microsoft_WizardLM-2-7B")
# model = AutoModelForCausalLM.from_pretrained("lucyknada/microsoft_WizardLM-2-7B")

# encoded_input = tokenizer('a  a a a a a a a a a frame the information as text status Not Returned ship id 12315 location Port Maine', return_tensors='pt')
# output = model.generate(**encoded_input, max_length=100)

# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)

# from transformers import BertGenerationConfig, BertGenerationEncoder

# configuration = BertGenerationConfig()
# model = BertGenerationEncoder(configuration)


# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# model = T5ForConditionalGeneration.from_pretrained("t5-base")
# input_ids = tokenizer.encode('summarize: shipment status Not Returned shipment id 1231 shipment location Bangkok', return_tensors='pt')
# greedy_output = model.generate(input_ids, num_beams=7, no_repeat_ngram_size=2, min_length=50, max_length=100)
# print("Output:\n" + 100 * '-')

# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

# model.generation_config
# input_ids = tokenizer.encode('The shipment status is Not Returned for the shipment id 1234', return_tensors='pt')

# output = model.generate(input_ids)
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)

# from transformers import pipeline

# text2text_generator = pipeline("text2text-generation")
# text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
# # [{'generated_text': 'the answer to life, the universe and everything'}]

# text2text_generator("translate from English to French: I'm very happy")
# # [{'generated_text': 'Je suis très heureux'}]

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
#print(tokenizer.decode(tokenized_chat[0]))

outputs = model.generate(tokenized_chat, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))