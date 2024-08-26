import numpy as np
import torch
import random
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding


MAX_LENGTH = 30     # max value of a text in the dataset; might need to recompute when dataset is larger
NUM_INTENTS = 3


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(dataset):
    return tokenizer(dataset["text"], truncation=True, max_length=MAX_LENGTH)


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    id2label = {0: 'Query_for_Shipment_Location', 1: 'Query_for_Shipment_StartEnd_Date', 2: 'Query_for_Shipments_sent_in_windowed_time_period'}
    label2id = {'Query_for_Shipment_Location': 0, 'Query_for_Shipment_StartEnd_Date': 1, 'Query_for_Shipments_sent_in_windowed_time_period': 2}
    
    # tokenization
    tokenizer = AutoTokenizer.from_pretrained('/Users/priyansha/zuum/venv3-9/src/distilbert-base-uncased')
    
    # retrieve test, train, and validate sets
    dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "val.csv"})
    
    print('{:>5,} training samples'.format(len(dataset["train"])))
    print('{:>5,} validation samples'.format(len(dataset["test"])))
    
    train_tokenized = dataset["train"].map(preprocess_function)
    val_tokenized = dataset["test"].map(preprocess_function, batched=True)
    accuracy = evaluate.load("accuracy")
    
    model = AutoModelForSequenceClassification.from_pretrained("/Users/priyansha/zuum/venv3-9/src/distilbert-base-uncased", 
                                                num_labels=NUM_INTENTS, id2label=id2label, label2id=label2id)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
                    output_dir="n_distilbert-base-uncased",
                    evaluation_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    num_train_epochs=2,
                    weight_decay=0.01,
                    push_to_hub=False,
                )

    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

    trainer.train()
    trainer.save_model()
    
    # IGNORE ------------------------>
    # # tokenize all the sentences
    # input_ids = []
    # attention_masks = []
    # labels = []

    # for i in range(dataframe.shape[0]):
    #     try:
    #         encoded_dict = tokenizer.encode_plus(
    #                         dataframe['text'][i],                      # Sentence to encode.
    #                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    #                         max_length = MAX_LENGTH,           # Pad & truncate all sentences.
    #                         truncation = True,
    #                         padding = 'max_length',
    #                         return_attention_mask = True,   # Construct attn. masks.
    #                         return_tensors = 'pt',     # Return pytorch tensors.
    #                 )
        
    #         # Add the encoded sentence to the list.    
    #         input_ids.append(encoded_dict['input_ids'])
        
    #         # And its attention mask (simply differentiates padding from non-padding).
    #         attention_masks.append(encoded_dict['attention_mask'])
    #         labels.append(labels_dict[dataframe['intent'][i]])

    #     except ValueError:
    #         continue

    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # labels_tensor = torch.tensor(labels)
    
    # dataset = TensorDataset(input_ids,attention_masks, labels_tensor)
    # batch_size = 32 # choose from 16 or 32

    # # Create the DataLoaders for our training and validation sets to save on memory
    # # We'll take training samples in random order. 
    # train_dataloader = DataLoader(
    #             train,  # The training samples.
    #             sampler = RandomSampler(train), # Select batches randomly
    #             batch_size = batch_size # Trains with this batch size.
    #         )

    # # For validation the order doesn't matter, so we'll just read them sequentially.
    # validation_dataloader = DataLoader(
    #             validation, # The validation samples.
    #             sampler = SequentialSampler(validation), # Pull out batches sequentially.
    #             batch_size = batch_size # Evaluate with this batch size.
    #         )
    
    # model = DistilBertModel.from_pretrained("/Users/priyansha/zuum/venv3-9/src/distilbert-base-uncased" 
    #                                         #num_labels = NUM_INTENTS,  
    #                                         #output_attentions = False, # Whether the model returns attentions weights.
    #                                         #output_hidden_states = False, # Whether the model returns all hidden-states.
    #                                         )

    # epochs = 2 # choose from 2, 3, or 4

    # # Total number of training steps is [number of batches] x [number of epochs]. 
    # # (Note that this is not the same as the number of training samples).
    # total_steps = len(train_dataloader) * epochs

    # optimizer = AdamW(model.parameters(),
    #               lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #               eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
    #             )

    # # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                             num_warmup_steps = 0, # Default value in run_glue.py
    #                                             num_training_steps = total_steps)
    

    # training_stats = []

    # # Measure the total training time for the whole run.
    # total_t0 = time.time()

    # # For each epoch...
    # for epoch_i in range(0, epochs):
    
    #     # ========================================
    #     #               Training
    #     # ========================================
        
    #     # Perform one full pass over the training set.

    #     print("")
    #     print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    #     print('Training...')

    #     # Measure how long the training epoch takes.
    #     t0 = time.time()

    #     # Reset the total loss for this epoch.
    #     total_train_loss = 0

    #     # Put the model into training mode. Don't be mislead--the call to 
    #     # `train` just changes the *mode*, it doesn't *perform* the training.
    #     # `dropout` and `batchnorm` layers behave differently during training
    #     # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    #     model.train()

    #     # For each batch of training data...
    #     for step, batch in enumerate(train_dataloader):

    #         # Progress update every 40 batches.
    #         if step % 40 == 0 and not step == 0:
    #             # Calculate elapsed time in minutes.
    #             elapsed = format_time(time.time() - t0)
                
    #             # Report progress.
    #             print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    #         # Unpack this training batch from our dataloader. x
    #         #
    #         # `batch` contains three pytorch tensors:
    #         #   [0]: input ids 
    #         #   [1]: attention masks
    #         #   [2]: labels 
                
    #         b_input_ids = batch[0].to(device)
    #         b_input_mask = batch[1].to(device)
    #         b_labels = batch[2].to(device)

    #         # Always clear any previously calculated gradients before performing a
    #         # backward pass. PyTorch doesn't do this automatically because 
    #         # accumulating the gradients is "convenient while training RNNs". 
    #         # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
    #         model.zero_grad()        

    #         # Perform a forward pass (evaluate the model on this training batch).
    #         # The documentation for this `model` function is here: 
    #         # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    #         # It returns different numbers of parameters depending on what arguments
    #         # arge given and what flags are set. For our useage here, it returns
    #         # the loss (because we provided labels) and the "logits"--the model
    #         # outputs prior to activation.
    #         loss, logits = model(b_input_ids, 
    #                             token_type_ids=None, 
    #                             attention_mask=b_input_mask, 
    #                             labels=b_labels)

    #         # Accumulate the training loss over all of the batches so that we can
    #         # calculate the average loss at the end. `loss` is a Tensor containing a
    #         # single value; the `.item()` function just returns the Python value 
    #         # from the tensor.
    #         print(loss.item())
    #         total_train_loss += loss.item()

    #         # Perform a backward pass to calculate the gradients.
    #         loss.backward()

    #         # Clip the norm of the gradients to 1.0.
    #         # This is to help prevent the "exploding gradients" problem.
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    #         # Update parameters and take a step using the computed gradient.
    #         # The optimizer dictates the "update rule"--how the parameters are
    #         # modified based on their gradients, the learning rate, etc.
    #         optimizer.step()

    #         # Update the learning rate.
    #         scheduler.step()

    #     # Calculate the average loss over all of the batches.
    #     avg_train_loss = total_train_loss / len(train_dataloader)            
        
    #     # Measure how long this epoch took.
    #     training_time = format_time(time.time() - t0)

    #     print("")
    #     print("  Average training loss: {0:.2f}".format(avg_train_loss))
    #     print("  Training epcoh took: {:}".format(training_time))
        
    #     # ========================================
    #     #               Validation
    #     # ========================================
    #     # After the completion of each training epoch, measure our performance on
    #     # our validation set.

    #     print("")
    #     print("Running Validation...")

    #     t0 = time.time()

    #     # Put the model in evaluation mode--the dropout layers behave differently
    #     # during evaluation.
    #     model.eval()

    #     # Tracking variables 
    #     total_eval_accuracy = 0
    #     total_eval_loss = 0
    #     nb_eval_steps = 0

    #     # Evaluate data for one epoch
    #     for batch in validation_dataloader:
            
    #         # Unpack this training batch from our dataloader.
    #         #
    #         # `batch` contains three pytorch tensors:
    #         #   [0]: input ids 
    #         #   [1]: attention masks
    #         #   [2]: labels 

    #         b_input_ids = batch[0].to(device)
    #         b_input_mask = batch[1].to(device)
    #         b_labels = batch[2].to(device)
        
    #         # Tell pytorch not to bother with constructing the compute graph during
    #         # the forward pass, since this is only needed for backprop (training).
            
    #         with torch.no_grad():        
                
    #             # Forward pass, calculate logit predictions.
    #             # token_type_ids is the same as the "segment ids", which 
    #             # differentiates sentence 1 and 2 in 2-sentence tasks.
    #             # The documentation for this `model` function is here: 
    #             # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    #             # Get the "logits" output by the model. The "logits" are the output
    #             # values prior to applying an activation function like the softmax.
    #             (loss, logits) = model(b_input_ids, 
    #                                 token_type_ids=None, 
    #                                 attention_mask=b_input_mask,
    #                                 labels=b_labels)
            
    #         # Accumulate the validation loss.
    #         total_eval_loss += loss.item()

    #         # Move logits and labels to CPU
    #         logits = logits.detach().cpu().numpy()
    #         label_ids = b_labels.to('cpu').numpy()

    #         # Calculate the accuracy for this batch of test sentences, and
    #         # accumulate it over all batches.
    #         total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    #     # Report the final accuracy for this validation run.
    #     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    #     print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    #     # Calculate the average loss over all of the batches.
    #     avg_val_loss = total_eval_loss / len(validation_dataloader)
        
    #     # Measure how long the validation run took.
    #     validation_time = format_time(time.time() - t0)
    
    #     print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    #     print("  Validation took: {:}".format(validation_time))

    #     # Record all statistics from this epoch.
    #     training_stats.append(
    #         {
    #             'epoch': epoch_i + 1,
    #             'Training Loss': avg_train_loss,
    #             'Valid. Loss': avg_val_loss,
    #             'Valid. Accur.': avg_val_accuracy,
    #             'Training Time': training_time,
    #             'Validation Time': validation_time
    #         }
    #     )

    # print("")
    # print("Training complete!")

    # print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))