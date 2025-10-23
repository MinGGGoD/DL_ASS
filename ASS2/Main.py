#!/usr/bin/env python
# coding: utf-8

# # <font color="#0b486b">  FIT5215: Deep Learning (2025) - Assignment 2 (Section I)</font>
# ***
# *CE/Lecturer (Clayton):*  **Dr Trung Le** | trunglm@monash.edu <br/>
# *Lecturer (Clayton):* **A/Prof Zongyuan Ge** | zongyuan.ge@monash.edu <br/>
# *Lecturer (Malaysia):*  **Dr Arghya Pal** | arghya.pal@monash.edu <br/>
#  <br/>
# *Head Tutor 3181:*  **Ms Ruda Nie H** |  \[RudaNie.H@monash.edu \] <br/>
# *Head Tutor 5215:*  **Ms Leila Mahmoodi** |  \[leila.mahmoodi@monash.edu \]
# 
# <br/> <br/>
# Faculty of Information Technology, Monash University, Australia
# ***

# # <font color="#0b486b">  Student Information</font>
# ***
# Surname: **Zhang**  <br/>
# Firstname: **Yiming**    <br/>
# Student ID: **35224436**    <br/>
# Email: **yzha1213@student.monash.edu**    <br/>
# Your tutorial time: **12pm Wed**    <br/>
# ***

# # <font color="0b486b">Assignment 2 – Deep Learning for Sequential Data</font>
# ### Due: <font color="red">11:55pm Sunday, 26 October 2025</font> (FIT5215)
# 
# #### <font color="red">Important note:</font> This is an **individual** assignment. It contributes **20%** to your final mark. Read the assignment instructions carefully.

# ## <font color="#0b486b">Assignment 2's Organization</font>
# This assignment 2 has two (2) sections:
# - Section 1: Fundamentals of RNNs (10 marks).
# - Section 2: Deep Learning for Sequential Data (90 marks). This section is further divided into 4 parts.
# 
# The assignment 2 is organized in three (3) notebooks.
# - Notebook 1 (this notebook) [Total: 30 marks] includes Section 1 as well as Part 1 and Part 2 of Section 2.
# - Notebook 2 ([link](https://colab.research.google.com/drive/1m0mh9Mk4-AKEhgAHRwQdl5mc0x7SF7Tv?usp=sharing)) [Total: 40 marks] includes Part 3 of Section 2.
# - Notebook 3 ([link](https://colab.research.google.com/drive/1JfMZeCkkvjZ5LvKNV-UnR10pl-RogMgF?usp=sharing)) [Total: 30 marks] includes Part 4 of Section 2.
# 

# ## <font color="#0b486b">What to submit</font>
# 
# This assignment is to be completed individually and submitted to Moodle unit site. **By the due date, you are required to submit one  <font color="red; font-weight:bold">single zip file, named xxx_assignment02_solution.zip</font> where `xxx` is your student ID, to the corresponding Assignment (Dropbox) in Moodle**. You can use Google Colab to do Assignment 2 but you need to save it to an `*.ipynb` file to submit to the unit Moodle.
# 
# **More importantly, if you use Google Colab to do this assignment, you need to first make a copy of this notebook on your Google drive**.

# ***For example, if your student ID is <font color="red; font-weight:bold">12356</font>, then gather all of your assignment solutions to a folder, create a zip file named <font color="red; font-weight:bold">123456_assignment02_solution.zip</font> and submit this file.***

# Within this zip folder, you **must** submit the following files <u>for each part</u>:
# 1.	**`FIT5215_DeepLearning_Assignment2_Official[Main].ipynb`**:  this is your Python notebook solution source file.
# 1.	**`FIT5215_DeepLearning_Assignment2_Official[Main].html`**: this is the output of your Python notebook solution *exported* in HTML format.
# 1. **`FIT5215_DeepLearning_Assignment2_Official[RNNs].ipynb`**
# 1. **`FIT5215_DeepLearning_Assignment2_Official[RNNs].html`**
# 1. **`FIT5215_DeepLearning_Assignment2_Official[Transformers].ipynb`**
# 1. **`FIT5215_DeepLearning_Assignment2_Official[Transformers].html`**
# 1.	Any **extra files or folder** needed to complete your assignment (e.g., images used in your answers).
# 
# 

# ## Section 1: Fundamentals in RNNs

# You need to **manually** implement a multi-timestep Recurrent Neural Network that can take an input as a 3D tensor `[batch_size, seq_len, input_size]` for a classification task.
# 
# <div style="text-align: right"><font color="red">[Total: 10 marks]</font></div>

# In[ ]:


import torch


# We declare the relevant variables.

# In[ ]:


input_size = 5
seq_len = 4
batch_size = 8
hidden_size = 3
num_classes = 3


# We create random inputs (i.e., `inputs`) and random labels (i.e., `random_labels`).

# In[ ]:


inputs = torch.randn(batch_size, seq_len, input_size)
random_labels = torch.randint(0, num_classes, (batch_size,))
print(inputs.shape)
print(random_labels)


# (1) In what follows, we need to declare the model parameters, which include the matrices $U$ (``[input_size, hidden_size]``), W (``[hidden_size, hidden_size]``), $V$ (``[hidden_size, num_classes]``) and the biases $b$ and $c$ for the hidden states and logits respectively.
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


#Insert your code here
U = torch.randn(input_size, hidden_size, requires_grad=True)
W = torch.randn(hidden_size, hidden_size, requires_grad=True)
b = torch.randn(hidden_size, requires_grad=True)
V = torch.randn(hidden_size, num_classes, requires_grad=True)
c = torch.randn(num_classes, requires_grad=True)


# (2) Next you need to write the code to compute `hiddens` which is a 3D tensor of the shape ``[batch_size, seq_len, hidden_size]`` using the formula of the simple/standard RNN cells. You can freely modify the code below.
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


#Initialize hiddens for update.
# hiddens = torch.zeros(batch_size, seq_len, hidden_size)

#Insert your code here
hiddens_list = []
for t in range(seq_len):
    if t == 0:
        # First timestep: h_t = tanh(U * x_t + b)
        h_t = torch.tanh(torch.matmul(inputs[:, t, :], U) + b)
        hiddens_list.append(h_t)
    else:
        # Subsequent timesteps: h_t = tanh(U * x_t + W * h_{t-1} + b)
        h_t = torch.tanh(torch.matmul(inputs[:, t, :], U) + torch.matmul(hiddens_list[t-1], W) + b)
        hiddens_list.append(h_t)
        
hiddens = torch.stack(hiddens_list, dim=1)

print(hiddens)


# (3) In what follows, you need to write the code to compute the logits based on the last hidden state (``[batch_size, hidden_size]``) of hiddens.
# 
# <div style="text-align: right"><font color="red">[1 mark]</font></div>

# In[ ]:


logits = torch.matmul(hiddens[:, -1, :], V) + c  # Use last hidden state
print(logits)


# (4) Write the code to compute the cross-entropy loss by comparing the logits to the labels. You can use PyTorch's built-in loss function.
# 
# <div style="text-align: right"><font color="red">[1 mark]</font></div>

# In[ ]:


#Insert your code here
loss = torch.nn.functional.cross_entropy(logits, random_labels)
print(loss)


# (5) Next, you need to do back-propagation to compute the gradients of the loss w.r.t. the model parameters. You can use PyTorch's built-in method to compute the gradients.
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


#Insert your code here
loss.backward()


# (6) Finally, let assume that the learning rate $\eta = 0.1$, you need to write the code to **manually** update the new model parameters using the SGD manner.
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


#Insert your code here
eta = 0.1
with torch.no_grad():
    U -= eta * U.grad
    W -= eta * W.grad
    b -= eta * b.grad
    V -= eta * V.grad
    c -= eta * c.grad
    
    # Zero gradients for next iteration
    U.grad.zero_()
    W.grad.zero_()
    b.grad.zero_()
    V.grad.zero_()
    c.grad.zero_()


# ## Section 2: Deep Learning for Sequential Data

# ### <font color="#0b486b">Set random seeds</font>

# We start with importing PyTorch and NumPy and setting random seeds for PyTorch and NumPy. You can use any seeds you prefer.

# In[ ]:


import os
import torch
import random
import requests
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import os
from six.moves.urllib.request import urlretrieve
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[ ]:


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_all(seed=1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## <font color="#0b486b">Download and preprocess the data</font>
# 
# <div style="text-align: right"><font color="red; font-weight:bold"><span></div>

# The dataset we use for this assignment is a question classification dataset for which the training set consists of $5,500$ questions belonging to 6 coarse question categories including:
# - abbreviation (ABBR),
# - entity (ENTY),
# - description (DESC),
# - human (HUM),
# - location (LOC) and
# - numeric (NUM).
# 
# In this assignment, we will utilize a subset of this dataset, containing $2,000$ questions for training and validation. We will use 80% of those 2000 questions for trainning and the rest for validation.
# 

# Preprocessing data is a crucial initial step in any machine learning or deep learning project. The *TextDataManager* class simplifies the process by providing functionalities to download and preprocess data specifically designed for the subsequent questions in this assignment. It is highly recommended to gain a comprehensive understanding of the class's functionality by **carefully reading** the content provided in the *TextDataManager* class before proceeding to answer the questions.

# In[ ]:


class DataManager:
    """
    This class manages and preprocesses a simple text dataset for a sentence classification task.

    Attributes:
        verbose (bool): Controls verbosity for printing information during data processing.
        max_sentence_len (int): The maximum length of a sentence in the dataset.
        str_questions (list): A list to store the string representations of the questions in the dataset.
        str_labels (list): A list to store the string representations of the labels in the dataset.
        numeral_labels (list): A list to store the numerical representations of the labels in the dataset.
        numeral_data (list): A list to store the numerical representations of the questions in the dataset.
        random_state (int): Seed value for random number generation to ensure reproducibility.
            Set this value to a specific integer to reproduce the same random sequence every time. Defaults to 6789.
        random (np.random.RandomState): Random number generator object initialized with the given random_state.
            It is used for various random operations in the class.

    Methods:
        maybe_download(dir_name, file_name, url, verbose=True):
            Downloads a file from a given URL if it does not exist in the specified directory.
            The directory and file are created if they do not exist.

        read_data(dir_name, file_names):
            Reads data from files in a directory, preprocesses it, and computes the maximum sentence length.
            Each file is expected to contain rows in the format "<label>:<question>".
            The labels and questions are stored as string representations.

        manipulate_data():
            Performs data manipulation by tokenizing, numericalizing, and padding the text data.
            The questions are tokenized and converted into numerical sequences using a tokenizer.
            The sequences are padded or truncated to the maximum sequence length.

        train_valid_test_split(train_ratio=0.9):
            Splits the data into training, validation, and test sets based on a given ratio.
            The data is randomly shuffled, and the specified ratio is used to determine the size of the training set.
            The string questions, numerical data, and numerical labels are split accordingly.
            TensorFlow `Dataset` objects are created for the training and validation sets.


    """

    def __init__(self, verbose=True, random_state=6789):
        self.verbose = verbose
        self.max_sentence_len = 0
        self.str_questions = list()
        self.str_labels = list()
        self.numeral_labels = list()
        self.maxlen = None
        self.numeral_data = list()
        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

    @staticmethod
    def maybe_download(dir_name, file_name, url, verbose=True):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(os.path.join(dir_name, file_name)):
            urlretrieve(url + file_name, os.path.join(dir_name, file_name))
        if verbose:
            print("Downloaded successfully {}".format(file_name))

    def read_data(self, dir_name, file_names):
        self.str_questions = list()
        self.str_labels = list()
        for file_name in file_names:
            file_path= os.path.join(dir_name, file_name)
            with open(file_path, "r", encoding="latin-1") as f:
                for row in f:
                    row_str = row.split(":")
                    label, question = row_str[0], row_str[1]
                    question = question.lower()
                    self.str_labels.append(label)
                    self.str_questions.append(question[0:-1])
                    if self.max_sentence_len < len(self.str_questions[-1]):
                        self.max_sentence_len = len(self.str_questions[-1])

        # turns labels into numbers
        le = preprocessing.LabelEncoder()
        le.fit(self.str_labels)
        self.numeral_labels = np.array(le.transform(self.str_labels))
        self.str_classes = le.classes_
        self.num_classes = len(self.str_classes)
        if self.verbose:
            print("\nSample questions and corresponding labels... \n")
            print(self.str_questions[0:5])
            print(self.str_labels[0:5])

    def manipulate_data(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab = self.tokenizer.get_vocab()
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i:w for w,i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        token_ids = []
        num_seqs = []
        for text in self.str_questions:  # iterate over the list of text
          text_seqs = self.tokenizer.tokenize(str(text))  # tokenize each text individually
          # Convert tokens to IDs
          token_ids = self.tokenizer.convert_tokens_to_ids(text_seqs)
          # Convert token IDs to a tensor of indices using your word2idx mapping
          seq_tensor = torch.LongTensor(token_ids)
          num_seqs.append(seq_tensor)  # append the tensor for each sequence

        # Pad the sequences and create a tensor
        if num_seqs:
          self.numeral_data = pad_sequence(num_seqs, batch_first=True)  # Pads to max length of the sequences
          self.num_sentences, self.maxlen = self.numeral_data.shape

    def train_valid_test_split(self, train_ratio=0.8, test_ratio = 0.1):
        train_size = int(self.num_sentences*train_ratio) +1
        test_size = int(self.num_sentences*test_ratio) +1
        valid_size = self.num_sentences - (train_size + test_size)
        data_indices = list(range(self.num_sentences))
        random.shuffle(data_indices)
        self.train_str_questions = [self.str_questions[i] for i in data_indices[:train_size]]
        self.train_numeral_labels = self.numeral_labels[data_indices[:train_size]]
        train_set_data = self.numeral_data[data_indices[:train_size]]
        train_set_labels = self.numeral_labels[data_indices[:train_size]]
        train_set_labels = torch.from_numpy(train_set_labels)
        train_set = torch.utils.data.TensorDataset(train_set_data, train_set_labels)
        self.test_str_questions = [self.str_questions[i] for i in data_indices[-test_size:]]
        self.test_numeral_labels = self.numeral_labels[data_indices[-test_size:]]
        test_set_data = self.numeral_data[data_indices[-test_size:]]
        test_set_labels = self.numeral_labels[data_indices[-test_size:]]
        test_set_labels = torch.from_numpy(test_set_labels)
        test_set = torch.utils.data.TensorDataset(test_set_data, test_set_labels)
        self.valid_str_questions = [self.str_questions[i] for i in data_indices[train_size:-test_size]]
        self.valid_numeral_labels = self.numeral_labels[data_indices[train_size:-test_size]]
        valid_set_data = self.numeral_data[data_indices[train_size:-test_size]]
        valid_set_labels = self.numeral_labels[data_indices[train_size:-test_size]]
        valid_set_labels = torch.from_numpy(valid_set_labels)
        valid_set = torch.utils.data.TensorDataset(valid_set_data, valid_set_labels)
        self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        self.valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)


# In[ ]:


print('Loading data...')
DataManager.maybe_download("data", "train_2000.label", "http://cogcomp.org/Data/QA/QC/")

dm = DataManager()
dm.read_data("data/", ["train_2000.label"])


# In[ ]:


dm.manipulate_data()
dm.train_valid_test_split(train_ratio=0.8, test_ratio = 0.1)


# In[ ]:


for x, y in dm.train_loader:
    print(x.shape, y.shape)
    break


# ## <font color="#0b486b">Part 1: Using Word2Vect to transform texts to vectors </font>
# 
# <div style="text-align: right"><font color="red; font-weight:bold">[Total marks for this part: 10 marks]<span></div>

# In[ ]:


import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np


# #### <font color="red">**Question 1.1**</font>
# **Write code to download the pretrained model *glove-wiki-gigaword-100*. Note that this model transforms a word in its dictionary to a $100$ dimensional vector.**
# 
# **Write code for the function *get_word_vector(word, model)* used to transform a word to a vector using the pretrained Word2Vect model *model*. Note that for a word not in the vocabulary of our *word2vect*, you need to return a vector $0$ with 100 dimensions.**
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


word2vect = api.load("glove-wiki-gigaword-100")


# In[ ]:


def get_word_vector(word, model):
    try:
        vector = model[word]
    except:
        vector = np.zeros(100)
    return vector


# #### <font color="red">**Question 1.2**</font>
# 
# **Write the code for the function `get_sentence_vector(sentence, important_score=None, model= None)`. Note that this function will transform a sentence to a 100-dimensional vector using the pretrained model *model*. In addition, the list *important_score* which has the same length as the *sentence* specifies the important scores of the words in the sentence. In your code, you first need to apply *softmax* function over *important_score* to obtain the important weight *important_weight* which forms a probability over the words of the sentence. Furthermore, the final vector of the sentence will be weighted sum of the individual vectors for words and the weights in *important_weight*.**
# - $important\_weight = softmax(important\_score)$.
# - $final\_vector= important\_weight[1]\times v[1] + important\_weight[2]\times v[2] + ...+ important\_weight[T]\times v[T]$ where $T$ is the length of the sentence and $v[i]$ is the vector representation of the $i-th$  word in this sentence.
# 
# **Note that if `important_score=None` is set by default, your function should return the average of all representation vectors corresponding to set `important_score=[1,1,...,1]`.**
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


def get_sentence_vector(sentence, important_score=None, model= None):
    #Insert your code here
    words = sentence.split()
    word_vectors = []
    
    for word in words:
        word_vector = get_word_vector(word, model)
        word_vectors.append(word_vector)
    
    if len(word_vectors) == 0:
        return np.zeros(100)
    
    word_vectors = np.array(word_vectors)
    
    if important_score is None:
        # Return average of all word vectors
        feature_vector = np.mean(word_vectors, axis=0)
    else:
        # Apply softmax to important_score to get weights
        important_weight = torch.softmax(torch.tensor(important_score, dtype=torch.float32), dim=0).numpy()
        # Weighted sum of word vectors
        feature_vector = np.sum(word_vectors * important_weight.reshape(-1, 1), axis=0)
    
    return feature_vector


# #### <font color="red">**Question 1.3**</font>
# 
# **Write code to transform questions in *dm.train_str_questions* and *dm.valid_str_questions* to feature vectors. Note that after running the following cells, you must have $X\_train$ and $X\_valid$ which are two NumPy arrays of the feature vectors and $y\_train$ and $y\_valid$ which are two arrays of numeric labels (Hint: *dm.train_numeral_labels* and *dm.valid_numeral_labels*). You can add more lines to the following cells if necessary. In addition, you should decide the *important_score* by yourself. For example, the 1st score is 1, the 2nd score is decayed by 0.9, the 3rd is decayed by 0.9, and so on.**
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


print("Transform training set to feature vectors...")
# Create important scores with decay factor 0.9
X_train = []
for sentence in dm.train_str_questions:
    words = sentence.split()
    important_score = [0.9 ** i for i in range(len(words))]
    feature_vector = get_sentence_vector(sentence, important_score, word2vect)
    X_train.append(feature_vector)
X_train = np.array(X_train)
y_train = dm.train_numeral_labels


# In[ ]:


print("Transform validation set to feature vectors...")
# Create important scores with decay factor 0.9
X_valid = []
for sentence in dm.valid_str_questions:
    words = sentence.split()
    important_score = [0.9 ** i for i in range(len(words))]
    feature_vector = get_sentence_vector(sentence, important_score, word2vect)
    X_valid.append(feature_vector)
X_valid = np.array(X_valid)
y_valid = dm.valid_numeral_labels


# #### <font color="red">**Question 1.4**</font>
# 
# **It is now to use *MinMaxScaler(feature_range=(-1,1))* in scikit-learn to scale both training and validation sets to the range $(-1,1)$.**
# 
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


#Insert your code
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
print(X_train.shape, X_valid.shape)


# #### <font color="red">**Question 1.5**</font>
# **Train a Logistic Regression model on the training set and then evaluate on the validation set.** You can use any classification metrics in `sklearn` for evaluation.
# <div style="text-align: right"><font color="red">[2 marks]</font></div>

# In[ ]:


from sklearn.linear_model import LogisticRegression
#Insert your code
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics
#Insert your code
y_pred = lr_model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(metrics.classification_report(y_valid, y_pred))


# We now declare the `BaseTrainer` class, which will be used later to train the subsequent deep learning models for text data.

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion  #the loss function
        self.optimizer = optimizer  #the optimizer
        self.train_loader = train_loader  #the train loader
        self.val_loader = val_loader  #the valid loader

    #the function to train the model in many epochs
    def fit(self, num_epochs):
        self.num_batches = len(self.train_loader)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()
            print(
                f'{self.num_batches}/{self.num_batches} - train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy*100:.4f}% \
                - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy*100:.4f}%')

    #train in one epoch, return the train_acc, train_loss
    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct / total
        train_loss = running_loss / self.num_batches
        return train_loss, train_accuracy

    #evaluate on a loader and return the loss and accuracy
    def evaluate(self, loader):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(self.val_loader)
        return loss, accuracy

    #return the val_acc, val_loss, be called at the end of each epoch
    def validate_one_epoch(self):
      val_loss, val_accuracy = self.evaluate(self.val_loader)
      return val_loss, val_accuracy


# ## <font color="#0b486b">Part 2: Text CNN for sequence modeling and neural embedding </font>
# 
# <div style="text-align: right"><font color="red; font-weight:bold">[Total marks for this part: 10 marks]<span></div>

# **In what follows, you are required to complete the code for Text CNN for sentence classification. The paper of Text CNN can be found at this [link](https://www.aclweb.org/anthology/D14-1181.pdf). Here is the description of the Text CNN that you need to construct.**
# - There are three attributes (properties or instance variables): *embed_size, state_size, data_manager*.
#   - `embed_size`: the dimension of the vector space for which the words are embedded to using the embedding matrix.
#   - `state_size`: the number of filters used in *Conv1D* (reference [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)).
#   - `data_manager`: the data manager to store information of the dataset.
# - The detail of the computational process is as follows:
#   - Given input $x$, we embed $x$ using the embedding matrix to obtain an $3D$ tensor $[batch\_size, seq\_len, embed\_size]$ as $e$.
#   - We feed $e$ to three *Conv1D* layers, each of which has $state\_size$ filters, activation= $relu$, and $kernel\_size= 3, 5, 7$ respectively to obtain $h1, h2, h3$. Note that each $h1, h2, h3$ is a 3D tensor with the shape $[batch\_size, state\_size, output\_size]$. Moreover, you need to apply *Conv1D* to the $seq\_len$ dimension.
#   - We then apply *GlobalMaxPool1D()* (reference [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html#torch.nn.functional.max_pool1d)) over $h1, h2, h3$ to obtain 2D tensors stored in $h1, h2, h3$ again.
#   - We then concatenate three 2D tensors $h1, h2, h3$ to obtain $h$ with the shape $\left[batch\_size, 3\times state\_size\right]$. Note that you need to specify the axis to concatenate.
#   - We finally build up one dense layer $\left[3\times state\_size, num\_classes\right]$  on the top of $h$ for classification.
#   

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

#You can modify the code if you want but need to keep the skeleton
class TextCNN(torch.nn.Module):
    def __init__(self, embed_size= 128, state_size=16, data_manager=None):
        super().__init__()
        self.data_manager = data_manager
        self.embed_size = embed_size
        self.state_size = state_size
        #declare the necessary layers here
        self.embed = nn.Embedding(self.data_manager.vocab_size, self.embed_size)
        self.conv1d_1 = nn.Conv1d(self.embed_size, self.state_size, kernel_size=3)
        self.conv1d_2 = nn.Conv1d(self.embed_size, self.state_size, kernel_size=5)
        self.conv1d_3 = nn.Conv1d(self.embed_size, self.state_size, kernel_size=7)
        self.fc = nn.Linear(state_size*3, self.data_manager.num_classes)

    def forward(self, x):
        e = self.embed(x)
        #permute x before applying Conv1D
        e= e.permute(0,2,1)

        #applying Conv1D
        h1 = F.relu(self.conv1d_1(e))
        h2 = F.relu(self.conv1d_2(e))
        h3 = F.relu(self.conv1d_3(e))

        #apply GlobalMaxPool
        h1 = F.max_pool1d(h1, kernel_size=h1.size(2))
        h2 = F.max_pool1d(h2, kernel_size=h2.size(2))
        h3 = F.max_pool1d(h3, kernel_size=h3.size(2))

        h = torch.cat([h1.squeeze(2), h2.squeeze(2), h3.squeeze(2)], dim=1)
        h = self.fc(h)
        return h


# We declare `text_cnn` and train on several epochs (e.g., `50 epochs`).
# 

# In[ ]:


text_cnn = TextCNN(data_manager=dm).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_cnn.parameters(), lr=0.001)
trainer = BaseTrainer(model=text_cnn, criterion=criterion, optimizer=optimizer, train_loader=dm.train_loader, val_loader=dm.valid_loader)
trainer.fit(num_epochs=50)


# We evaluate the trained model on the testing set.

# In[ ]:


test_loss, test_acc = trainer.evaluate(dm.test_loader)
print(f'test_loss: {test_loss:.4f} - test_accuracy: {test_acc*100:.4f}%')

