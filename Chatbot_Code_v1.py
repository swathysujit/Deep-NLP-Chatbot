# Building a Chat Bot with Deep NLP using Tensorflow

# Data from link :https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
# Metadata - not used in training, just additional info
# We need only the conversations between characters - so taking only movie_conversations and movie_lines files from the data

# Importing the dataset
import numpy as np
import tensorflow as tf
import re
import time

########################Part 1 - Data Preprocessing###################################

# Importing dataset
lines = open("movie_lines.txt",encoding = "utf-8",errors = "ignore").read().split('\n')
conversations = open("movie_conversations.txt",encoding = "utf-8",errors = "ignore").read().split('\n')

# Creating a dictionary that maps each line to its ID
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
            
# Creating a list of all conversations
conversations_ids = []
for conversation in conversations[:-1]:
    #split, take the last part(column using -1, then choose 2nd to 2nd last character since we do not want square brackets
    #Replace ' with nothing and spaces with nothing. Now we have a cleaned list of conversation ids
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))

# Getting separately the questions and answers - questions - input to NN and answers - Ouput of NN
# We want to separate huge lists of Q and A but of same size, to have one to one mapping
# In the lines, 1st line will be Question, second line will be answer    
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"[-()\"#$/@;:<>{}+-=~|.?,]","",text)
    return text

# Cleaning the questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its frequency/number of occurrences
# We want only have essential words, lets say we can remove the words that occur less than 5% of the times

word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

# Creating two dictionaries that map the questions words and the answers words to a unique integer
# Here we will add an if condition to check if the word frequency is above a certain hreshold or not, if it is not, this word will not be included in the dictionary
# Tokenization here is mapping a unique integer to each Q nd A

# Choosing threshold (recommended to filter out 5% of the least occuring corpus words)
threshold = 20
questionswords2int = {}
word_number = 0
for word,count in word2count.items():
    if count>=threshold:
        questionswords2int[word]=word_number
        word_number+=1
        
answerswords2int = {}
word_number = 0
for word,count in word2count.items():
    if count>=threshold:
        answerswords2int[word]=word_number
        word_number+=1
    
# Adding the last tokens to these two dictionaries
# These are needed for the encoder and decoder
# PAD(very important for the model) the seq and model shouls all have same lengths, therefore we insert this in an empty position 
# All words that have been filtered out by our two dictionaries(5% least freq words are replaced laterw wiht OUT)
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1

for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1

# Creating the inverse dictionary of the answerswords2int dictionary 
# We will use this inverse mapping of integers to answers words, while building the Seq2Seq model
# Learn this simple trick, its important and useful! 
answersints2word = {w_i:w for w,w_i in answerswords2int.items()}

# Adding the End of String token to the end of every answer (EOS is needed in the end of decoding layer)
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all the questions and the answers into integers
# And replacing all the words we filtered out by <OUT> token
# We are doing this so that we can sort all the questions and answers by their length, this is to optimize training performance

questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)

# Sorting questions and answers by the length of questions
# This will speed up the training, reduce the loss, it will increase the padding
sorted_clean_questions = []
sorted_clean_answers = []

# We want to include only Q and A that are not too long, for better learning. Say 25 length set here, can change as per convenience
for length in range(1,25+1  ):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            #To have the question and answers mapped, we will append the answer for the corresponding question
            sorted_clean_answers.append(answers_to_int[i[0]])
            
            
###################################### Part-2 Deep NLP : Building the Seq2Seq Model ################################################################################## 

# Creating placeholders for the inputs and the targets
# Tensors - advanced array, allows faster computations in deep NNs. So we need to move from numpy arrays to tensors
# This is the first step we need to do always when working on deep models with Tensorflow

def model_inputs():
    inputs = tf.placeholder(tf.int32,[None,None],name = 'input')
    targets = tf.placeholder(tf.int32,[None,None],name = 'target')
    #Drop out - amount of neurons to deactivate during an iteration, this is done by the keep_prob parameter which controls the drop out rate
    lr = tf.placeholder(tf.float32,name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
    return inputs,targets,lr,keep_prob

# Preprocessing the Targets
# Since the decoders will only process a special format of targets - target must be in batches, the NN does not accept single answers, so we will feed the NN with batches o some answers say 10 at a time    
# Second imp element is that each of the answers in the targets must start with the <SOS> token, so we need to put the SOS(unique integer encoding this token) into each batch
# Since we need to keep same size for padding, we will take all except last column and concatenate the SOS token in the beginning
    
def preprocess_targets(targets,word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1]) #Decoder will not use the last token, so we are taking all from SOS to second last token
    preprocessed_targets = tf.concat([left_side,right_side],axis=1) #Horizontal concatenations = axix 1, vertical is axis = 0
    return preprocessed_targets

# Creating the encoding layer of the Seq2Seq model : The Encoder RNN Layer ! 
# This will be a stacked LSTM using basic LSTM cell class of tensorflow (instead of GRU), in which we apply drop out to improve performance
#rnn_inputs = model_inputs we created earlier, rnn_size is not number of layers in rnn, it is number of input tensors in the rnn layer we are creating
# num_layers is number of layers in rnn, keep_prob is for drop out and improve accuracy, seq_length = list of the length of each question in the batch
def encoder_rnn(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        #now we are getting same lstm, but with the dropout applied - technique of eactivating % of nuerons during iterations
        #we'll use dropout wrapper class of tensorflow for this
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        # We are ready to create the encoder cell, using the multi-rnn cell class of tensorflow
        # The encoder cell is composed of many lstm layers(we applied dropout to them), hence we multiply it with the number of layers we intend to have 
        encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        # Now we will get the encoder state from the bidirectional dynamix rnn function by the 'NN module' of tensorflow
        # we can add underscore and coma to indicate that we need only the encoder state, second output from the tf class
        encoder_ouput, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,sequence_length=sequence_length,inputs = rnn_inputs,dtype=tf.float32) #This dynamic version will take input and builds independent forward and backward rnn, but we have to make sure input cell and output cell must be same size
        return encoder_state


# Decoding the training set
# Decoder Embedding - Mapping from words to vectors of real numbers each one encoding uniquely the words associated wit it
# This is the format the decoder takes as inputs
# Decoding Scope -> Variable scope - an advanced data structure that will wrap the tensorflow variables        

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    # Attention keys are to be compared with target states, the attention values are used to create the context vectors,attention_score_function is used to compute the similarities between the keys and the target states,attention_construct_fuction is used to build the attention state 
    attention_keys, attention_values, attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option='bahdanau',num_units = decoder_cell.output_size)
    # Next we need to get the training decoder function
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,name = "attn_dec_train")
    #What we did here is we got an attentional decoder function for our dynamic rnn decoder. this func is used in the future while creating rnn decoder
    #Below, we need only the first element returned, the second and third is not needed, just replace with underscore
    decoder_output,decoder_final_state,decoder_final_context = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,training_decoder_function,decoder_embedded_input,sequence_length,scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validation set
# This function will be used to answer the questions asked in the test and also for cross validation during the validation steps
# We will do some cross validation to reduce overfitting
# Previously we used the attention_decoder_fn_train function, now we need to use the test function, so we will use the attention_decoder_fn_inference function ->     
# We added four new arguments as needed by inference func in this function 
    
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix,sos_id,eos_id,maximum_length,num_words,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    # Attention keys are to be compared with target states, the attention values are used to create the context vectors,attention_score_function is used to compute the similarities between the keys and the target states,attention_construct_fuction is used to build the attention state 
    attention_keys, attention_values, attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option='bahdanau',num_units = decoder_cell.output_size)
    # Next we need to get the training decoder function
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,encoder_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,decoder_embeddings_matrix,sos_id,eos_id,maximum_length,num_words,name = "attn_dec_inf")
    #What we did here is we got an attentional decoder function for our dynamic rnn decoder. this func is used in the future while creating rnn decoder
    #Below, we need only the first element returned, the second and third is not needed, just replace with underscore
    test_predictions,_,_ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,test_decoder_function,scope = decoding_scope)
    #We dont need dropout here, since drop out is only used in training to improve accuracy, not in test
    return test_predictions

# Creating the Decoder RNN :
# 
def decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,encoder_state,num_words,sequence_length,rnn_size,num_layers,word2int,keep_prob,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        #creating stacked lstm layers
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        # Initializing the fully connected weights
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        # Initializing biases with zeroes
        biases = tf.zeros_initializer()
        # We need to make the fully connected layer using the output function
        # Keeping default activation function - RLU while building the fully connected network
        # Normalizer = None, we wont do normalization
        output_function = lambda x: tf.contrib.layers.fully_connected(x,num_words,None,scope = decoding_scope,weights_initializer = weights,biases_initializer = biases)
        # Getting the training predictions, using the decode_training_set function we created before
        training_predictions = decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,decoder_cell,decoder_embeddings_matrix,word2int['<SOS>'],word2int['<EOS>'],sequence_length-1,num_words,sequence_length,decoding_scope,output_function,keep_prob,batch_size)
    return training_predictions,test_predictions

# Building the final Seq2Seq model - putting together encoder ad decoder parts of the architecture
# This is the function we are going to chat with in the test phase, that is, this is the brain of the chat bot

#Assembing the encoder and decoder    
def seq2seq_model(inputs, targets,keep_prob, batch_size, sequence_length, answers_num_words,questions_num_words,encoder_embedding_size, decoder_embedding_size,rnn_size,num_layers,questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, answers_num_words+1, encoder_embedding_size,initializer = tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input,rnn_size, num_layers, keep_prob,sequence_length)
    preprocessed_targets = preprocess_targets(targets,questionswords2int,batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix,encoder_state,questions_num_words,sequence_length,rnn_size,num_layers,questionswords2int,keep_prob,batch_size)
    return training_predictions, test_predictions    

################################### PART - 3 : Training  the Seq2Seq Model #################################################################33
# Setting the Hyperparameters        
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01 # Should not be too high or too low, either model will learn too fast, or too slow and not speak properly
# Check the code in 'Checkpoint' section in udemy to finalize the hypreparameters
learning_rate_decay = 0.9
min_learning_rate = 0.0001
# learning rate decay decides by which percentage the learnng rate is reduced in each iteration, so that it can learn in depth the logic of conversations, this is very important, we are choosing a common value 90%
# Also we need to set minimum learning rate, since we have many iterations and we do not want the learning rate to be tooo small
# We will also add early stopping, so that the training ends early, to avoid over-fitting
# Drop out rate also ensures no over-fitting happens, but while testing we wont keep drop out, all neurons participate in the prediction, only in training we apply drop out
# We are choosing drop-out of 50% which means keep-prob = 50&(1-drop-out)
keep_probability = 0.5
#We are done with hyperparameters, we can add more hyperparameters, but for starters this should be fine!

# Defining a tensorflow session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs (Seq2Seg model inputs)
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
# We are going to set a defualt value when the output is not fed into the rnn, so we use placeholder with default function
sequence_length = tf.placeholder_with_default(25,None,name = 'sequence_length')# We chose max length of 25 above, now this sets the maximum length of questions and answers
# So in training we won't be using Q and A more than length 25 words

# Getting the shape of the input tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
# This is not training ye, that will come later when we loop around epochs, this is required before that, based on inputs

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]),targets,keep_prob,batch_size,sequence_length,len(answerswords2int),len(questionswords2int),encoding_embedding_size,decoding_embedding_size,rnn_size,num_layers,questionswords2int)

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets, tf.ones([input_shape[0],sequence_length]))
    #Using adam optimizer, and apply gradient clipping to it
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.),grad_variable) for grad_tensor,grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    

# Padding the sequences  with the  <PAD> token sine we want all Q and A sequence to be of the same length, therefore we will apply padding        
# Example Question: ['Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>]
# Example Answer : [<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>, <PAD>]

def apply_padding(batch_of_sequences, word2int):
    # Maximum size of all sequences in a batch
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence+[word2int['<PAD>']]*(max_sequence_length-len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers
# Inputs - Questions, Targets - Answers
# Divide by '//' return integer result    
def split_into_batches(questions,answers,batch_size):
    for batch_index in range(0,len(questions)//batch_size):
        start_index = batch_index*batch_size
        questions_in_batch = questions[start_index:start_index+batch_size]
        answers_in_batch = answers[start_index:start_index+batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch,questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch,answerswords2int))
        # 'yield' in python is like return, but it is better to use yield when returning sequences!
        yield padded_questions_in_batch, padded_answers_in_batch

# Splitting the questions and answers into training and validation sets
# Performing cross validation by keeping 10-15% of training dataset aside and testing the model alongside while training
# Getting the index which will split the first 15% of the questions index and the last
training_validation_split = int(len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
        
############################### Training #####################################################3
batch_index_check_training_loss = 100        
batch_index_check_validation_loss = ((len(training_questions))//batch_size//2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000 #This is large since we want to have the training go through all 100 epochs
checkpoint = "C:/Users/Swathy Sujit/Documents/DNLP Chat Bot/chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())

for epoch in range(1):
    for batch_index,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batches(training_questions,training_answers,batch_size)):
        starting_time = time.time()
        _,batch_training_loss_error = session.run([optimizer_gradient_clipping,loss_error],{inputs: padded_questions_in_batch,targets:padded_answers_in_batch,lr:learning_rate,sequence_length:padded_answers_in_batch.shape[1], keep_prob:keep_probability})
        total_training_loss_error +=batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print ('Epoch : {:>3}/{}, Batch : {:>4}/{},Training Loss Error: {:>6.3f},Training Time on 100 Batches,{:d} seconds'.format(epoch,epochs,batch_index,len(training_questions)//batch_size,total_training_loss_error/batch_index_check_training_loss,int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index>0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batches(validation_questions,validation_answers,batch_size)):
                batch_validation_loss_error = session.run(loss_error,{inputs: padded_questions_in_batch,targets:padded_answers_in_batch,lr:learning_rate,sequence_length:padded_answers_in_batch.shape[1], keep_prob: 1 }) #No need keep probability for validation, so we set keep_prob to 1 always
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error/(len(validation_questions)/batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error,int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error < min(list_validation_loss_error):
                print("I speak better now!!")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session,checkpoint)
            else:
                print("Sorry I do not speak better, I need to practise more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")

saver = tf.train.Saver()
saver.save(session,checkpoint)

######################################### Part 4 : Testing ##############################################################
# Using weights from the github repository

# Loading the weights and running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session,checkpoint)
    
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question,word2int):
    question = clean_text(question)
    return [word2int.get(word,word2int['<OUT>']) for word in question.split()]

# Setting up the chat
while(True):
    question = input("You : ")
    if question == "Goodbye":
        break
    question = convert_string2int(question,questionswords2int)
    question = question + [questionswords2int['<PAD>']]*(20-len(question))
    #Neural network acceots only a input of batches, so we need to input it as a batch, hence we intriduce a fake batch variable
    fake_batch = np.zeros((batch_size,20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions,{inputs:fake_batch,keep_prob:0.5})[0]
    # We need to post - process it and clean it
    answer = ''
    for i in np.argmax(predicted_answer,1):
        if answersints2word[i] == 'i':
            token = 'I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token == ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
    
        