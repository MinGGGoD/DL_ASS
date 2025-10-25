# <font color="#0b486b">  Student Information</font>
***
Surname: **Zhang**  <br/>
Firstname: **Yiming**    <br/>
Student ID: **35224436**    <br/>
Email: **yzha1213@student.monash.edu**    <br/>
Your tutorial time: **12pm Wed**    <br/>
***

## Section 2: Deep Learning for Sequential Data

### <font color="#0b486b">Set random seeds</font>

We need to install the package datasets for creating BERT datasets.


```python
# !pip install datasets
!pip install datasets==4.0.0
!pip install transformers==4.57.0
```

We start with importing PyTorch and NumPy and setting random seeds for PyTorch and NumPy. You can use any seeds you prefer.


```python
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
```


```python
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
```

## <font color="#0b486b">Download and preprocess the data</font>

<div style="text-align: right"><font color="red; font-weight:bold"><span></div>

The dataset we use for this assignment is a question classification dataset for which the training set consists of $5,500$ questions belonging to 6 coarse question categories including:
- abbreviation (ABBR),
- entity (ENTY),
- description (DESC),
- human (HUM),
- location (LOC) and
- numeric (NUM).

In this assignment, we will utilize a subset of this dataset, containing $2,000$ questions for training and validation. We will use 80% of those 2000 questions for trainning and the rest for validation.


Preprocessing data is a crucial initial step in any machine learning or deep learning project. The *TextDataManager* class simplifies the process by providing functionalities to download and preprocess data specifically designed for the subsequent questions in this assignment. It is highly recommended to gain a comprehensive understanding of the class's functionality by **carefully reading** the content provided in the *TextDataManager* class before proceeding to answer the questions.


```python
class DataManager:
    """
    This class manages and preprocesses a simple text dataset for a sentence classification task.

    Attributes:
        verbose (bool): Controls verbosity for printing information during data processing.
        max_sentence_len (int): The maximum length of a sentence in the dataset.
        str_questions (list): A list to store the string representations of the questions in the dataset.
        str_labels (list): A list to store the string representations of the labels in the dataset.
        numeral_labels (list): A list to store the numerical representations of the labels in the dataset.
        maxlen (int): Maximum length for padding sequences. Sequences longer than this length will be truncated,
            and sequences shorter than this length will be padded with zeros. Defaults to 50.
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
          self.num_sentences, self.max_seq_len = self.numeral_data.shape

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
```


```python
print('Loading data...')
DataManager.maybe_download("data", "train_2000.label", "http://cogcomp.org/Data/QA/QC/")

dm = DataManager()
dm.read_data("data/", ["train_2000.label"])
```

    Loading data...
    Downloaded successfully train_2000.label
    
    Sample questions and corresponding labels... 
    
    ['manner how did serfdom develop in and then leave russia ?', 'cremat what films featured the character popeye doyle ?', "manner how can i find a list of celebrities ' real names ?", 'animal what fowl grabs the spotlight after the chinese year of the monkey ?', 'exp what is the full form of .com ?']
    ['DESC', 'ENTY', 'DESC', 'ENTY', 'ABBR']



```python
dm.manipulate_data()
dm.train_valid_test_split(train_ratio=0.8, test_ratio = 0.1)
```


```python
for x, y in dm.train_loader:
    print(x.shape, y.shape)
    break
```

    torch.Size([64, 36]) torch.Size([64])


We now declare the `BaseTrainer` class, which will be used later to train the subsequent deep learning models for text data.


```python
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
```

## <font color="#0b486b">Part 4: Transformer-based models for sequence modeling and neural embedding</font>

<div style="text-align: right"><font color="red; font-weight:bold">[Total marks for this part: 30 marks]<span></div>

#### <font color="red">**Question 4.1**</font>

**Implement the multi-head attention module of the Transformer for the text classification problem. The provided code is from our tutorial. In this part, we only use the output of the Transformer encoder for the classification task. For further information on the Transformer model, refer to [this paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).**

<div style="text-align: right"><font color="red; font-weight:bold">[Total marks for this part: 10 marks]<span></div>


Below is the code of `MultiHeadSelfAttention`, `PositionWiseFeedForward`, `PositionalEncoding`, and `EncoderLayer`.


```python
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def scaled_dot_product_attention(self, Q, K, V):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        #if mask is not None:
            #attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
```


```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```


```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```


```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

Your task is to develop `TransformerClassifier` in which we input the embedding with the shape `[batch_size, seq_len, embed_dim]` to some `EncoderLayer` (i.e., num_layers specifies the number of EncoderLayer) and then compute the average of all token embeddings (i.e., `[batch_size, seq_len, embed_dim]`) across the `seq_len`. Finally, on the top of this average embedding, we build up a linear layer for making predictions.


```python
class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.2, data_manager = None):
        super(TransformerClassifier, self).__init__()
        self.vocab_size = data_manager.vocab_size
        self.num_classes = data_manager.num_classes
        self.embed_dim = embed_dim
        self.max_seq_len = data_manager.max_seq_len
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    def build(self):
        #Insert your code here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # positional encoding
        self.pos_encoding = PositionalEncoding(self.embed_dim, self.max_seq_len)
        
        # encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        
        # output layer
        self.output_layer = nn.Linear(self.embed_dim, self.num_classes)


    def forward(self, x):
        #Insert your code here
        embedded = self.embedding(x)
        
        # add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # pass through encoder layers
        output = embedded
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
            
        # average pooling across the sequence length
        output = torch.mean(output, dim=1)
        
        return self.output_layer(output)



```


```python
transformer = TransformerClassifier(embed_dim=512, num_heads=8, ff_dim=2048, num_layers=12, dropout_rate=0.1, data_manager= dm)
transformer.build()
transformer = transformer.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
trainer = BaseTrainer(model= transformer, criterion=criterion, optimizer=optimizer, train_loader=dm.train_loader, val_loader=dm.valid_loader)
trainer.fit(num_epochs=30)

```

    Epoch 1/30
    26/26 - train_loss: 2.0195 - train_accuracy: 22.4235%                 - val_loss: 0.8066 - val_accuracy: 12.6263%
    Epoch 2/30
    26/26 - train_loss: 1.7231 - train_accuracy: 18.8007%                 - val_loss: 0.8273 - val_accuracy: 12.6263%
    Epoch 3/30
    26/26 - train_loss: 1.6846 - train_accuracy: 21.9238%                 - val_loss: 0.8405 - val_accuracy: 26.2626%
    Epoch 4/30
    26/26 - train_loss: 1.6877 - train_accuracy: 21.9863%                 - val_loss: 0.7704 - val_accuracy: 27.2727%
    Epoch 5/30
    26/26 - train_loss: 1.6989 - train_accuracy: 20.8620%                 - val_loss: 0.7289 - val_accuracy: 27.2727%
    Epoch 6/30
    26/26 - train_loss: 1.6803 - train_accuracy: 21.3616%                 - val_loss: 0.7095 - val_accuracy: 27.2727%
    Epoch 7/30
    26/26 - train_loss: 1.6395 - train_accuracy: 26.2961%                 - val_loss: 0.7330 - val_accuracy: 37.3737%
    Epoch 8/30
    26/26 - train_loss: 1.3738 - train_accuracy: 39.1006%                 - val_loss: 0.7387 - val_accuracy: 31.3131%
    Epoch 9/30
    26/26 - train_loss: 1.3327 - train_accuracy: 37.6015%                 - val_loss: 0.6473 - val_accuracy: 45.9596%
    Epoch 10/30
    26/26 - train_loss: 1.3249 - train_accuracy: 39.9750%                 - val_loss: 0.6861 - val_accuracy: 45.9596%
    Epoch 11/30
    26/26 - train_loss: 1.3189 - train_accuracy: 40.2873%                 - val_loss: 0.6670 - val_accuracy: 36.8687%
    Epoch 12/30
    26/26 - train_loss: 1.3511 - train_accuracy: 40.2249%                 - val_loss: 0.6711 - val_accuracy: 45.9596%
    Epoch 13/30
    26/26 - train_loss: 1.3152 - train_accuracy: 38.9756%                 - val_loss: 0.6618 - val_accuracy: 45.9596%
    Epoch 14/30
    26/26 - train_loss: 1.3364 - train_accuracy: 40.2873%                 - val_loss: 0.6684 - val_accuracy: 45.9596%
    Epoch 15/30
    26/26 - train_loss: 1.2674 - train_accuracy: 39.5378%                 - val_loss: 0.6410 - val_accuracy: 48.9899%
    Epoch 16/30
    26/26 - train_loss: 1.2080 - train_accuracy: 43.2854%                 - val_loss: 0.6185 - val_accuracy: 52.0202%
    Epoch 17/30
    26/26 - train_loss: 1.2382 - train_accuracy: 40.9119%                 - val_loss: 0.6292 - val_accuracy: 52.0202%
    Epoch 18/30
    26/26 - train_loss: 1.2265 - train_accuracy: 42.4110%                 - val_loss: 0.6471 - val_accuracy: 42.9293%
    Epoch 19/30
    26/26 - train_loss: 1.2401 - train_accuracy: 43.7227%                 - val_loss: 0.6808 - val_accuracy: 52.0202%
    Epoch 20/30
    26/26 - train_loss: 1.2177 - train_accuracy: 42.6608%                 - val_loss: 0.6830 - val_accuracy: 52.0202%
    Epoch 21/30
    26/26 - train_loss: 1.2324 - train_accuracy: 41.7239%                 - val_loss: 0.6623 - val_accuracy: 52.0202%
    Epoch 22/30
    26/26 - train_loss: 1.2359 - train_accuracy: 40.6621%                 - val_loss: 0.6318 - val_accuracy: 52.0202%
    Epoch 23/30
    26/26 - train_loss: 1.1551 - train_accuracy: 42.0362%                 - val_loss: 2.0134 - val_accuracy: 35.3535%
    Epoch 24/30
    26/26 - train_loss: 1.2195 - train_accuracy: 50.5934%                 - val_loss: 1.0479 - val_accuracy: 43.4343%
    Epoch 25/30
    26/26 - train_loss: 0.9268 - train_accuracy: 54.2161%                 - val_loss: 0.2734 - val_accuracy: 71.7172%
    Epoch 26/30
    26/26 - train_loss: 0.6357 - train_accuracy: 66.7083%                 - val_loss: 0.2279 - val_accuracy: 71.2121%
    Epoch 27/30
    26/26 - train_loss: 0.4715 - train_accuracy: 76.0775%                 - val_loss: 0.1273 - val_accuracy: 83.3333%
    Epoch 28/30
    26/26 - train_loss: 0.4462 - train_accuracy: 78.4510%                 - val_loss: 0.3573 - val_accuracy: 73.2323%
    Epoch 29/30
    26/26 - train_loss: 0.4136 - train_accuracy: 80.5746%                 - val_loss: 0.1292 - val_accuracy: 82.3232%
    Epoch 30/30
    26/26 - train_loss: 0.3259 - train_accuracy: 88.6946%                 - val_loss: 0.1456 - val_accuracy: 90.4040%


#### <font color="red">**Question 4.2**</font>
**Prefix prompt-tuning with Transformers: You need to implement the prefix prompt-tuning with Transformers. Basically, we base on a pre-trained Transformer, add prefix prompts, and do fine-tuning for a target dataset.**

<div style="text-align: right"><font color="red; font-weight:bold">[Total marks for this part: 10 marks]<span></div>

To implement prefix prompt-tuning with pretrained Transformers, we first need to create the Bert dataset.


```python
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from datasets import Dataset

model_name = "bert-base-uncased"  # BERT or any similar model

# Tokenize input and prepare model inputs
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = Dataset.from_dict({"text": dm.str_questions, "label": dm.numeral_labels})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length= 36)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print(dataset)
```


    Map:   0%|          | 0/2000 [00:00<?, ? examples/s]


    Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 2000
    })


The following function splits the BERT dataset `dataset` into three BERT datasets for training, valid, and testing.


```python
def train_valid_test_split(dataset, train_ratio=0.8, test_ratio = 0.1):
    num_sentences = len(dataset)
    train_size = int(num_sentences*train_ratio) +1
    test_size = int(num_sentences*test_ratio) +1
    valid_size = num_sentences - (train_size + test_size)
    train_set = dataset[:train_size]
    train_set = Dataset.from_dict(train_set)
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_set = dataset[-test_size:]
    test_set = Dataset.from_dict(test_set)
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    valid_set = dataset[train_size:-test_size]
    valid_set = Dataset.from_dict(valid_set)
    valid_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
    return train_loader, test_loader, valid_loader

```


```python
train_loader, test_loader, valid_loader = train_valid_test_split(dataset)
```

You need to implement the class `PrefixTuningForClassification` for the prefix prompt fine-tuning. We first load a pre-trained BERT model specified by `model_name`. The parameter `prefix_length` specifies the length of the prefix prompts we add to the pre-trained BERT model. Specifically, given the input batch `[batch_size, seq_len]`, we input to the embedding layer of the pre-trained BERT model to obtain `[batch_size, seq_len, embed_size]`. We create the prefix prompts $P$ of the size `[prefix_length, embed_size]` and concatenate to the embeddings from the pre-trained BERT to obtain `[batch_size, seq_len + prefix_length, embed_size]`. This concatenation tensor will then be fed to the encoder layers of the pre-trained BERT layer to obtain the last `[batch_size, seq_len + prefix_length, embed_size]`.

We then take mean across the seq_len to obtain `[batch_size, embed_size]` on which we can build up a linear layer for making predictions. Please note that **the parameters to tune include the prefix prompts $P$** and **the output linear layer**, and you should freeze the parameters of the BERT pre-trained model. Moreover, your code should cover the edge case when `prefix_length=None`. In this case, we do not insert any prefix prompts and we only do fine-tuning for the output linear layer on top.  


```python
class PrefixTuningForClassification(nn.Module):
    def __init__(self, model_name, prefix_length=None, data_manager = None):
        super(PrefixTuningForClassification, self).__init__()

        # Load the pretrained transformer model (BERT-like model)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.hidden_size =  self.model.config.hidden_size
        self.prefix_length = prefix_length
        self.num_classes = data_manager.num_classes
        # Insert your code here
        if self.prefix_length is not None:
            self.prefix_embeddings = nn.Parameter(torch.randn(self.prefix_length, self.hidden_size))
            
        # freeze the pre-trained parameters of the BERT model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # add a new classification head
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask):
        # Insert your code here
        embeddings = self.model.embeddings(input_ids)
        
        # add prefix embeddings if specified
        if self.prefix_length is not None:
            batch_size = embeddings.size(0)
            prefix_embeddings = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1) # repeat?
            embeddings = torch.cat([prefix_embeddings, embeddings], dim=1)
            # update the attention mask
            prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
            
        # pass through the encode layers
        output = self.model.encoder(embeddings, attention_mask=attention_mask.unsqueeze(1).unsqueeze(2))
        
        # use the last hidden state for classification
        if self.prefix_length is not None:
            last_hidden_state = output.last_hidden_state[:, self.prefix_length:, :]
        else:
            last_hidden_state = output.last_hidden_state
            
        # average pooling
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # do classification
        return self.classifier(pooled_output)

```

You can use the following `FineTunedBaseTrainer` to train the prompt fine-tuning models.


```python
class FineTunedBaseTrainer:
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
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            self.optimizer.zero_grad()
            outputs = self.model(input_ids= input_ids, attention_mask= attention_mask)
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
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = self.model(input_ids= input_ids, attention_mask= attention_mask)
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
```

We declare and train the prefix-prompt tuning model. In addition, you need to be patient with this model because it might converge slowly with many epochs.


```python
prefix_tuning_model = PrefixTuningForClassification(model_name = "bert-base-uncased", prefix_length = 5, data_manager = dm).to(device)
```


```python
if prefix_tuning_model.prefix_length is not None:
  optimizer = torch.optim.Adam(list(prefix_tuning_model.classifier.parameters()) + [prefix_tuning_model.prefix_embeddings], lr=5e-5)
else:
  optimizer = torch.optim.Adam(prefix_tuning_model.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
trainer = FineTunedBaseTrainer(model= prefix_tuning_model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=valid_loader)
trainer.fit(num_epochs=100)
```

    Epoch 1/100
    26/26 - train_loss: 1.7964 - train_accuracy: 22.2361%                 - val_loss: 0.9475 - val_accuracy: 22.7273%
    Epoch 2/100
    26/26 - train_loss: 1.7408 - train_accuracy: 23.9225%                 - val_loss: 0.9086 - val_accuracy: 25.2525%
    Epoch 3/100
    26/26 - train_loss: 1.6994 - train_accuracy: 25.0468%                 - val_loss: 0.8821 - val_accuracy: 26.2626%
    Epoch 4/100
    26/26 - train_loss: 1.6880 - train_accuracy: 26.1087%                 - val_loss: 0.8636 - val_accuracy: 25.7576%
    Epoch 5/100
    26/26 - train_loss: 1.6604 - train_accuracy: 26.0462%                 - val_loss: 0.8442 - val_accuracy: 28.2828%
    Epoch 6/100
    26/26 - train_loss: 1.6598 - train_accuracy: 25.6090%                 - val_loss: 0.8355 - val_accuracy: 25.7576%
    Epoch 7/100
    26/26 - train_loss: 1.6545 - train_accuracy: 26.3585%                 - val_loss: 0.8310 - val_accuracy: 28.2828%
    Epoch 8/100
    26/26 - train_loss: 1.6338 - train_accuracy: 29.6690%                 - val_loss: 0.8239 - val_accuracy: 28.7879%
    Epoch 9/100
    26/26 - train_loss: 1.6297 - train_accuracy: 27.7327%                 - val_loss: 0.8216 - val_accuracy: 29.7980%
    Epoch 10/100
    26/26 - train_loss: 1.6095 - train_accuracy: 29.6065%                 - val_loss: 0.8097 - val_accuracy: 31.3131%
    Epoch 11/100
    26/26 - train_loss: 1.6268 - train_accuracy: 32.6046%                 - val_loss: 0.8146 - val_accuracy: 31.3131%
    Epoch 12/100
    26/26 - train_loss: 1.6067 - train_accuracy: 32.3548%                 - val_loss: 0.8068 - val_accuracy: 32.8283%
    Epoch 13/100
    26/26 - train_loss: 1.6014 - train_accuracy: 37.5390%                 - val_loss: 0.7940 - val_accuracy: 34.3434%
    Epoch 14/100
    26/26 - train_loss: 1.5931 - train_accuracy: 36.0400%                 - val_loss: 0.8019 - val_accuracy: 31.8182%
    Epoch 15/100
    26/26 - train_loss: 1.5883 - train_accuracy: 35.6027%                 - val_loss: 0.8006 - val_accuracy: 35.3535%
    Epoch 16/100
    26/26 - train_loss: 1.5972 - train_accuracy: 33.3542%                 - val_loss: 0.8001 - val_accuracy: 33.8384%
    Epoch 17/100
    26/26 - train_loss: 1.5820 - train_accuracy: 35.7277%                 - val_loss: 0.7952 - val_accuracy: 36.3636%
    Epoch 18/100
    26/26 - train_loss: 1.5695 - train_accuracy: 38.4135%                 - val_loss: 0.7818 - val_accuracy: 36.8687%
    Epoch 19/100
    26/26 - train_loss: 1.5836 - train_accuracy: 39.0381%                 - val_loss: 0.7842 - val_accuracy: 40.9091%
    Epoch 20/100
    26/26 - train_loss: 1.5632 - train_accuracy: 38.6633%                 - val_loss: 0.7887 - val_accuracy: 38.8889%
    Epoch 21/100
    26/26 - train_loss: 1.5639 - train_accuracy: 36.2898%                 - val_loss: 0.7881 - val_accuracy: 35.3535%
    Epoch 22/100
    26/26 - train_loss: 1.5583 - train_accuracy: 36.2274%                 - val_loss: 0.7963 - val_accuracy: 34.8485%
    Epoch 23/100
    26/26 - train_loss: 1.5672 - train_accuracy: 35.9775%                 - val_loss: 0.7966 - val_accuracy: 34.8485%
    Epoch 24/100
    26/26 - train_loss: 1.5605 - train_accuracy: 36.7270%                 - val_loss: 0.7903 - val_accuracy: 35.8586%
    Epoch 25/100
    26/26 - train_loss: 1.5477 - train_accuracy: 39.1630%                 - val_loss: 0.7817 - val_accuracy: 39.8990%
    Epoch 26/100
    26/26 - train_loss: 1.5457 - train_accuracy: 43.6602%                 - val_loss: 0.7656 - val_accuracy: 39.3939%
    Epoch 27/100
    26/26 - train_loss: 1.5485 - train_accuracy: 42.8482%                 - val_loss: 0.7689 - val_accuracy: 42.4242%
    Epoch 28/100
    26/26 - train_loss: 1.5388 - train_accuracy: 42.5359%                 - val_loss: 0.7684 - val_accuracy: 43.9394%
    Epoch 29/100
    26/26 - train_loss: 1.5352 - train_accuracy: 45.0344%                 - val_loss: 0.7561 - val_accuracy: 42.9293%
    Epoch 30/100
    26/26 - train_loss: 1.5383 - train_accuracy: 44.7845%                 - val_loss: 0.7608 - val_accuracy: 43.9394%
    Epoch 31/100
    26/26 - train_loss: 1.5180 - train_accuracy: 43.4728%                 - val_loss: 0.7650 - val_accuracy: 44.9495%
    Epoch 32/100
    26/26 - train_loss: 1.5164 - train_accuracy: 42.8482%                 - val_loss: 0.7730 - val_accuracy: 42.9293%
    Epoch 33/100
    26/26 - train_loss: 1.5311 - train_accuracy: 40.8495%                 - val_loss: 0.7684 - val_accuracy: 40.9091%
    Epoch 34/100
    26/26 - train_loss: 1.5305 - train_accuracy: 43.9725%                 - val_loss: 0.7648 - val_accuracy: 45.9596%
    Epoch 35/100
    26/26 - train_loss: 1.5110 - train_accuracy: 45.4716%                 - val_loss: 0.7602 - val_accuracy: 47.9798%
    Epoch 36/100
    26/26 - train_loss: 1.4996 - train_accuracy: 46.4085%                 - val_loss: 0.7474 - val_accuracy: 45.4545%
    Epoch 37/100
    26/26 - train_loss: 1.5045 - train_accuracy: 47.0956%                 - val_loss: 0.7500 - val_accuracy: 46.9697%
    Epoch 38/100
    26/26 - train_loss: 1.5073 - train_accuracy: 45.7214%                 - val_loss: 0.7437 - val_accuracy: 44.9495%
    Epoch 39/100
    26/26 - train_loss: 1.5024 - train_accuracy: 48.4072%                 - val_loss: 0.7471 - val_accuracy: 48.4848%
    Epoch 40/100
    26/26 - train_loss: 1.4994 - train_accuracy: 49.4066%                 - val_loss: 0.7537 - val_accuracy: 52.5253%
    Epoch 41/100
    26/26 - train_loss: 1.5007 - train_accuracy: 45.9088%                 - val_loss: 0.7586 - val_accuracy: 46.9697%
    Epoch 42/100
    26/26 - train_loss: 1.5898 - train_accuracy: 48.1574%                 - val_loss: 0.7535 - val_accuracy: 51.5152%
    Epoch 43/100
    26/26 - train_loss: 1.5013 - train_accuracy: 47.6577%                 - val_loss: 0.7503 - val_accuracy: 52.0202%
    Epoch 44/100
    26/26 - train_loss: 1.4680 - train_accuracy: 50.9681%                 - val_loss: 0.7516 - val_accuracy: 54.5455%
    Epoch 45/100
    26/26 - train_loss: 1.4801 - train_accuracy: 49.9688%                 - val_loss: 0.7496 - val_accuracy: 50.5051%
    Epoch 46/100
    26/26 - train_loss: 1.4754 - train_accuracy: 49.2192%                 - val_loss: 0.7487 - val_accuracy: 51.5152%
    Epoch 47/100
    26/26 - train_loss: 1.4657 - train_accuracy: 46.9706%                 - val_loss: 0.7470 - val_accuracy: 50.0000%
    Epoch 48/100
    26/26 - train_loss: 1.4581 - train_accuracy: 46.6583%                 - val_loss: 0.7435 - val_accuracy: 48.9899%
    Epoch 49/100
    26/26 - train_loss: 1.4561 - train_accuracy: 49.9063%                 - val_loss: 0.7333 - val_accuracy: 51.5152%
    Epoch 50/100
    26/26 - train_loss: 1.4435 - train_accuracy: 51.6552%                 - val_loss: 0.7250 - val_accuracy: 50.5051%
    Epoch 51/100
    26/26 - train_loss: 1.4649 - train_accuracy: 51.0931%                 - val_loss: 0.7286 - val_accuracy: 52.0202%
    Epoch 52/100
    26/26 - train_loss: 1.4476 - train_accuracy: 50.2186%                 - val_loss: 0.7349 - val_accuracy: 51.0101%
    Epoch 53/100
    26/26 - train_loss: 1.4689 - train_accuracy: 48.5322%                 - val_loss: 0.7333 - val_accuracy: 52.0202%
    Epoch 54/100
    26/26 - train_loss: 1.4462 - train_accuracy: 50.9057%                 - val_loss: 0.7317 - val_accuracy: 54.0404%
    Epoch 55/100
    26/26 - train_loss: 1.4340 - train_accuracy: 53.9663%                 - val_loss: 0.7164 - val_accuracy: 54.5455%
    Epoch 56/100
    26/26 - train_loss: 1.4663 - train_accuracy: 53.4041%                 - val_loss: 0.7062 - val_accuracy: 53.5354%
    Epoch 57/100
    26/26 - train_loss: 1.4501 - train_accuracy: 51.4678%                 - val_loss: 0.7171 - val_accuracy: 54.5455%
    Epoch 58/100
    26/26 - train_loss: 1.4359 - train_accuracy: 54.4035%                 - val_loss: 0.7181 - val_accuracy: 59.5960%
    Epoch 59/100
    26/26 - train_loss: 1.4282 - train_accuracy: 53.4666%                 - val_loss: 0.7125 - val_accuracy: 57.5758%
    Epoch 60/100
    26/26 - train_loss: 1.4162 - train_accuracy: 57.2142%                 - val_loss: 0.7140 - val_accuracy: 60.1010%
    Epoch 61/100
    26/26 - train_loss: 1.4195 - train_accuracy: 55.7152%                 - val_loss: 0.7171 - val_accuracy: 57.5758%
    Epoch 62/100
    26/26 - train_loss: 1.4188 - train_accuracy: 54.0287%                 - val_loss: 0.7205 - val_accuracy: 54.5455%
    Epoch 63/100
    26/26 - train_loss: 1.4073 - train_accuracy: 54.4660%                 - val_loss: 0.7217 - val_accuracy: 57.5758%
    Epoch 64/100
    26/26 - train_loss: 1.4093 - train_accuracy: 55.1530%                 - val_loss: 0.7264 - val_accuracy: 58.5859%
    Epoch 65/100
    26/26 - train_loss: 1.3945 - train_accuracy: 52.0300%                 - val_loss: 0.7294 - val_accuracy: 57.0707%
    Epoch 66/100
    26/26 - train_loss: 1.4081 - train_accuracy: 51.7177%                 - val_loss: 0.7285 - val_accuracy: 56.5657%
    Epoch 67/100
    26/26 - train_loss: 1.4227 - train_accuracy: 51.6552%                 - val_loss: 0.7254 - val_accuracy: 56.5657%
    Epoch 68/100
    26/26 - train_loss: 1.3860 - train_accuracy: 55.9650%                 - val_loss: 0.7156 - val_accuracy: 59.0909%
    Epoch 69/100
    26/26 - train_loss: 1.3912 - train_accuracy: 54.7158%                 - val_loss: 0.7215 - val_accuracy: 58.5859%
    Epoch 70/100
    26/26 - train_loss: 1.3947 - train_accuracy: 56.1524%                 - val_loss: 0.7115 - val_accuracy: 61.1111%
    Epoch 71/100
    26/26 - train_loss: 1.3813 - train_accuracy: 57.5265%                 - val_loss: 0.7097 - val_accuracy: 58.0808%
    Epoch 72/100
    26/26 - train_loss: 1.3838 - train_accuracy: 56.7146%                 - val_loss: 0.7105 - val_accuracy: 59.0909%
    Epoch 73/100
    26/26 - train_loss: 1.3784 - train_accuracy: 56.1524%                 - val_loss: 0.7078 - val_accuracy: 58.0808%
    Epoch 74/100
    26/26 - train_loss: 1.3782 - train_accuracy: 54.7783%                 - val_loss: 0.7072 - val_accuracy: 56.0606%
    Epoch 75/100
    26/26 - train_loss: 1.3941 - train_accuracy: 55.4653%                 - val_loss: 0.7042 - val_accuracy: 56.5657%
    Epoch 76/100
    26/26 - train_loss: 1.3699 - train_accuracy: 58.5259%                 - val_loss: 0.6983 - val_accuracy: 61.6162%
    Epoch 77/100
    26/26 - train_loss: 1.3652 - train_accuracy: 60.3998%                 - val_loss: 0.6841 - val_accuracy: 61.1111%
    Epoch 78/100
    26/26 - train_loss: 1.3634 - train_accuracy: 58.2136%                 - val_loss: 0.6963 - val_accuracy: 63.1313%
    Epoch 79/100
    26/26 - train_loss: 1.3680 - train_accuracy: 57.0893%                 - val_loss: 0.6938 - val_accuracy: 60.1010%
    Epoch 80/100
    26/26 - train_loss: 1.3607 - train_accuracy: 60.2748%                 - val_loss: 0.6898 - val_accuracy: 62.6263%
    Epoch 81/100
    26/26 - train_loss: 1.3514 - train_accuracy: 59.9001%                 - val_loss: 0.6770 - val_accuracy: 62.6263%
    Epoch 82/100
    26/26 - train_loss: 1.3472 - train_accuracy: 58.7758%                 - val_loss: 0.6880 - val_accuracy: 63.1313%
    Epoch 83/100
    26/26 - train_loss: 1.3651 - train_accuracy: 57.5265%                 - val_loss: 0.6923 - val_accuracy: 61.6162%
    Epoch 84/100
    26/26 - train_loss: 1.3371 - train_accuracy: 59.0256%                 - val_loss: 0.6919 - val_accuracy: 59.5960%
    Epoch 85/100
    26/26 - train_loss: 1.3401 - train_accuracy: 57.7139%                 - val_loss: 0.6909 - val_accuracy: 60.1010%
    Epoch 86/100
    26/26 - train_loss: 1.3517 - train_accuracy: 58.4635%                 - val_loss: 0.6907 - val_accuracy: 59.5960%
    Epoch 87/100
    26/26 - train_loss: 1.3516 - train_accuracy: 59.3379%                 - val_loss: 0.6838 - val_accuracy: 59.5960%
    Epoch 88/100
    26/26 - train_loss: 1.3475 - train_accuracy: 60.7121%                 - val_loss: 0.6875 - val_accuracy: 59.5960%
    Epoch 89/100
    26/26 - train_loss: 1.3372 - train_accuracy: 59.4004%                 - val_loss: 0.6886 - val_accuracy: 63.6364%
    Epoch 90/100
    26/26 - train_loss: 1.3291 - train_accuracy: 60.5247%                 - val_loss: 0.6734 - val_accuracy: 64.6465%
    Epoch 91/100
    26/26 - train_loss: 1.3380 - train_accuracy: 59.7751%                 - val_loss: 0.6712 - val_accuracy: 64.6465%
    Epoch 92/100
    26/26 - train_loss: 1.3404 - train_accuracy: 63.0231%                 - val_loss: 0.6702 - val_accuracy: 64.6465%
    Epoch 93/100
    26/26 - train_loss: 1.3172 - train_accuracy: 62.8357%                 - val_loss: 0.6732 - val_accuracy: 66.1616%
    Epoch 94/100
    26/26 - train_loss: 1.3108 - train_accuracy: 62.0237%                 - val_loss: 0.6705 - val_accuracy: 67.1717%
    Epoch 95/100
    26/26 - train_loss: 1.3070 - train_accuracy: 62.8982%                 - val_loss: 0.6708 - val_accuracy: 66.1616%
    Epoch 96/100
    26/26 - train_loss: 1.3042 - train_accuracy: 62.4610%                 - val_loss: 0.6574 - val_accuracy: 64.1414%
    Epoch 97/100
    26/26 - train_loss: 1.3114 - train_accuracy: 61.2742%                 - val_loss: 0.6639 - val_accuracy: 64.6465%
    Epoch 98/100
    26/26 - train_loss: 1.2868 - train_accuracy: 61.9613%                 - val_loss: 0.6612 - val_accuracy: 68.1818%
    Epoch 99/100
    26/26 - train_loss: 1.2826 - train_accuracy: 62.7733%                 - val_loss: 0.6530 - val_accuracy: 66.1616%
    Epoch 100/100
    26/26 - train_loss: 1.2958 - train_accuracy: 63.2105%                 - val_loss: 0.6597 - val_accuracy: 67.1717%


#### <font color="red">**Question 4.3**</font>
**For any models defined in the previous questions (of all parts), you are free to fine-tune hyperparameters, e.g., `optimizer`, `learning_rate`, `state_sizes`, such that you get a best model, i.e., the one with the highest accuracy on the test set. You will need to report (i) what is your best model,  (ii) its accuracy on the test set, and (iii) the values of its hyperparameters. Note that you must report your best model's accuracy with rounding to 4 decimal places, i.e., 0.xxxx. You will also need to upload your best model (or provide us with the link to download your best model). The assessment will be based on your best model's accuracy, with up to 10 marks available, specifically:**
* The best accuracy $\ge$ 0.97: 10 marks
* 0.97 $>$ The best accuracy $\ge$ 0.92: 7 marks
* 0.92 $>$ The best accuracy $\ge$ 0.85: 4 marks
* The best accuracy $<$ 0.85: 0 mark

**For this question, you can put below the code to train the best model. In this case, you need to show your code and the evidence of running regarding the best model. Moreover, if you save the best model, you need to provide the link to download the best model, the code to load the best model, and then evaluate on the test set.**
<div style="text-align: right"><font color="red">[10 marks]</font></div>

##### Import Packages


```python
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
```

##### Define DistilBERT Classifier


```python
class DistilBERTClassifier(nn.Module):
    """
    Classifier using DistilBERT
    """
    def __init__(self, model_name="distilbert-base-uncased", num_classes=6, 
                 dropout_rate=0.3, hidden_dim=256):
        super(DistilBERTClassifier, self).__init__()
        
        # Load DistilBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.model_name = model_name
        
        # Freeze all layers
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 transformer layers
        for layer in self.bert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Compact classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
    
    def save_model(self, path):
        """
        Save only fine-tuned layers
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {}
        
        # Save last 2 transformer layers
        for i, layer in enumerate(self.bert.transformer.layer[-2:]):
            state_dict[f'transformer_layer_{i}'] = layer.state_dict()
        
        # Save classification head
        state_dict['classifier'] = self.classifier.state_dict()
        
        # Save metadata
        state_dict['_metadata'] = {
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'num_classes': len(self.classifier[-1].weight),
            'hidden_dim': self.classifier[0].out_features,
            'dropout_rate': self.classifier[3].p
        }
        
        torch.save(state_dict, path)
        
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ Model saved!")
        print(f"  Path: {path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        
    def load_best_model(self, path):
        """Load fine-tuned layers"""
        state_dict = torch.load(path, map_location=device)
        
        # Load transformer layers
        for i, layer in enumerate(self.bert.transformer.layer[-2:]):
            layer.load_state_dict(state_dict[f'transformer_layer_{i}'])
        
        # Load classifier
        self.classifier.load_state_dict(state_dict['classifier'])
        
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ Model loaded! Size: {file_size_mb:.2f} MB")
```

##### Define My Trainer with Early Stopping


```python
class MyTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_loader, 
                 val_loader, test_loader, patience=10, checkpoint_dir='.'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.patience_counter = 0
        self.best_epoch = 0
        
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})
        
        return running_loss / len(self.train_loader), correct / total
    
    def evaluate(self, loader, desc='Evaluating'):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return running_loss / len(loader), correct / total
    
    def fit(self, num_epochs):
        print(f"\nTraining model for {num_epochs} epochs")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print("-" * 80)
            
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader, 'Validation')
            test_loss, test_acc = self.evaluate(self.test_loader, 'Testing')
            
            print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
            print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.4f}%')
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_test_acc = test_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save model
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                self.model.save_model(checkpoint_path)
                
                # Save metadata
                metadata_path = os.path.join(self.checkpoint_dir, 'training_metadata.pt')
                torch.save({
                    'epoch': epoch,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                }, metadata_path)
                
                print(f'✓ Best model saved! Val: {val_acc*100:.2f}%, Test: {test_acc*100:.4f}%')
            else:
                self.patience_counter += 1
                print(f'No improvement for {self.patience_counter} epoch(s)')
                
                if self.patience_counter >= self.patience:
                    print(f'\nEarly stopping at epoch {epoch + 1}')
                    break
        
        print("\n" + "=" * 80)
        print(f'Best Model:')
        print(f'  - Epoch: {self.best_epoch}')
        print(f'  - Val Acc: {self.best_val_acc*100:.2f}%')
        print(f'  - Test Acc: {self.best_test_acc:.4f}')
        print("=" * 80)
        
        return self.best_test_acc
```

##### Data Preparation Function


```python
def prepare_data(dm, model_name="distilbert-base-uncased", max_length=48, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = Dataset.from_dict({
        "text": dm.str_questions, 
        "label": dm.numeral_labels
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    num_samples = len(dataset)
    train_size = int(num_samples * 0.8)
    test_size = int(num_samples * 0.1)
    val_size = num_samples - train_size - test_size
    
    train_set = Dataset.from_dict(dataset[:train_size])
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    val_set = Dataset.from_dict(dataset[train_size:train_size+val_size])
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    test_set = Dataset.from_dict(dataset[-test_size:])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    return train_loader, val_loader, test_loader
```

##### Train the Best Model


```python
# Hyperparameters
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
NUM_EPOCHS = 50
DROPOUT = 0.3
HIDDEN_DIM = 256
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 48
CHECKPOINT_DIR = '.'  # Save in current directory

print("=" * 80)
print("TRAINING MODEL (DistilBERT)")
print("=" * 80)
print(f"\nModel: {MODEL_NAME}")
print(f"  - Size: ~66M parameters")
print(f"  - Layers: 6")
print(f"  - Expected file size: ~80MB")

print(f"\nHyperparameters:")
print(f"  - LR: {LEARNING_RATE}")
print(f"  - Batch: {BATCH_SIZE}")
print(f"  - Dropout: {DROPOUT}")
print(f"  - Checkpoint: {CHECKPOINT_DIR}")

# Prepare data
print("\nPreparing data...")
train_loader, val_loader, test_loader = prepare_data(dm, MODEL_NAME, MAX_LENGTH, BATCH_SIZE)

# Initialize model
print("\nInitializing model...")
model = DistilBERTClassifier(
    model_name=MODEL_NAME,
    num_classes=dm.num_classes,
    dropout_rate=DROPOUT,
    hidden_dim=HIDDEN_DIM
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# Optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# Train
trainer = MyTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    patience=10,
    checkpoint_dir=CHECKPOINT_DIR
)

best_test_acc = trainer.fit(NUM_EPOCHS)

print(f"\n✓ Training completed!")
print(f"✓ Best test accuracy: {best_test_acc:.4f}")
print(f"✓ Model saved to: {CHECKPOINT_DIR}/best_model.pt")
```

    ================================================================================
    TRAINING MODEL (DistilBERT)
    ================================================================================
    
    Model: distilbert-base-uncased
      - Size: ~66M parameters
      - Layers: 6
      - Expected file size: ~80MB
    
    Hyperparameters:
      - LR: 3e-05
      - Batch: 32
      - Dropout: 0.3
      - Checkpoint: .
    
    Preparing data...



    Map:   0%|          | 0/2000 [00:00<?, ? examples/s]


    Train: 1600 | Val: 200 | Test: 200
    
    Initializing model...
      Total params: 66,561,798
      Trainable params: 14,374,662
    
    Training model for 50 epochs
    ================================================================================
    
    Epoch 1/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.48it/s, loss=1.7048, acc=25.00%]
                                                             

    
    Train Loss: 1.7557 | Train Acc: 25.00%
    Val Loss: 1.5499 | Val Acc: 44.50%
    Test Loss: 1.5665 | Test Acc: 43.0000%
    ✓ Model saved!
      Path: ./best_model.pt
      Size: 54.85 MB
    ✓ Best model saved! Val: 44.50%, Test: 43.0000%
    
    Epoch 2/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.49it/s, loss=1.0458, acc=55.69%]
                                                             

    
    Train Loss: 1.3368 | Train Acc: 55.69%
    Val Loss: 0.8136 | Val Acc: 84.50%
    Test Loss: 0.8940 | Test Acc: 77.5000%
    ✓ Model saved!
      Path: ./best_model.pt
      Size: 54.85 MB
    ✓ Best model saved! Val: 84.50%, Test: 77.5000%
    
    Epoch 3/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.44it/s, loss=0.3118, acc=86.81%]
                                                             

    
    Train Loss: 0.6255 | Train Acc: 86.81%
    Val Loss: 0.2687 | Val Acc: 95.50%
    Test Loss: 0.2940 | Test Acc: 96.5000%
    ✓ Model saved!
      Path: ./best_model.pt
      Size: 54.85 MB
    ✓ Best model saved! Val: 95.50%, Test: 96.5000%
    
    Epoch 4/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.40it/s, loss=0.1330, acc=95.81%]
                                                             

    
    Train Loss: 0.2300 | Train Acc: 95.81%
    Val Loss: 0.0908 | Val Acc: 99.00%
    Test Loss: 0.0918 | Test Acc: 98.5000%
    ✓ Model saved!
      Path: ./best_model.pt
      Size: 54.85 MB
    ✓ Best model saved! Val: 99.00%, Test: 98.5000%
    
    Epoch 5/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.41it/s, loss=0.1305, acc=98.44%]
                                                             

    
    Train Loss: 0.0977 | Train Acc: 98.44%
    Val Loss: 0.0420 | Val Acc: 100.00%
    Test Loss: 0.0414 | Test Acc: 99.5000%
    ✓ Model saved!
      Path: ./best_model.pt
      Size: 54.85 MB
    ✓ Best model saved! Val: 100.00%, Test: 99.5000%
    
    Epoch 6/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.39it/s, loss=0.0414, acc=99.12%]
                                                             

    
    Train Loss: 0.0577 | Train Acc: 99.12%
    Val Loss: 0.0266 | Val Acc: 100.00%
    Test Loss: 0.0318 | Test Acc: 99.0000%
    No improvement for 1 epoch(s)
    
    Epoch 7/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.31it/s, loss=0.0909, acc=99.75%]
                                                             

    
    Train Loss: 0.0313 | Train Acc: 99.75%
    Val Loss: 0.0253 | Val Acc: 99.00%
    Test Loss: 0.0349 | Test Acc: 99.0000%
    No improvement for 2 epoch(s)
    
    Epoch 8/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.36it/s, loss=0.0168, acc=99.88%] 
                                                             

    
    Train Loss: 0.0221 | Train Acc: 99.88%
    Val Loss: 0.0190 | Val Acc: 99.50%
    Test Loss: 0.0276 | Test Acc: 99.5000%
    No improvement for 3 epoch(s)
    
    Epoch 9/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.35it/s, loss=0.0115, acc=100.00%]
                                                             

    
    Train Loss: 0.0163 | Train Acc: 100.00%
    Val Loss: 0.0158 | Val Acc: 100.00%
    Test Loss: 0.0202 | Test Acc: 99.5000%
    No improvement for 4 epoch(s)
    
    Epoch 10/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.34it/s, loss=0.0127, acc=100.00%]
                                                             

    
    Train Loss: 0.0138 | Train Acc: 100.00%
    Val Loss: 0.0171 | Val Acc: 99.50%
    Test Loss: 0.0266 | Test Acc: 99.5000%
    No improvement for 5 epoch(s)
    
    Epoch 11/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.33it/s, loss=0.0093, acc=100.00%]
                                                             

    
    Train Loss: 0.0124 | Train Acc: 100.00%
    Val Loss: 0.0109 | Val Acc: 100.00%
    Test Loss: 0.0227 | Test Acc: 99.5000%
    No improvement for 6 epoch(s)
    
    Epoch 12/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.33it/s, loss=0.0101, acc=100.00%]
                                                             

    
    Train Loss: 0.0104 | Train Acc: 100.00%
    Val Loss: 0.0089 | Val Acc: 100.00%
    Test Loss: 0.0222 | Test Acc: 99.5000%
    No improvement for 7 epoch(s)
    
    Epoch 13/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.36it/s, loss=0.0085, acc=100.00%]
                                                             

    
    Train Loss: 0.0096 | Train Acc: 100.00%
    Val Loss: 0.0232 | Val Acc: 99.00%
    Test Loss: 0.0319 | Test Acc: 99.5000%
    No improvement for 8 epoch(s)
    
    Epoch 14/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.35it/s, loss=0.0104, acc=100.00%]
                                                             

    
    Train Loss: 0.0088 | Train Acc: 100.00%
    Val Loss: 0.0083 | Val Acc: 100.00%
    Test Loss: 0.0242 | Test Acc: 99.5000%
    No improvement for 9 epoch(s)
    
    Epoch 15/50
    --------------------------------------------------------------------------------


    Training: 100%|██████████| 50/50 [00:04<00:00, 12.38it/s, loss=0.0073, acc=100.00%]
                                                             

    
    Train Loss: 0.0082 | Train Acc: 100.00%
    Val Loss: 0.0173 | Val Acc: 99.00%
    Test Loss: 0.0268 | Test Acc: 99.5000%
    No improvement for 10 epoch(s)
    
    Early stopping at epoch 15
    
    ================================================================================
    Best Model:
      - Epoch: 5
      - Val Acc: 100.00%
      - Test Acc: 0.9950
    ================================================================================
    
    ✓ Training completed!
    ✓ Best test accuracy: 0.9950
    ✓ Model saved to: ./best_model.pt


    

##### Test the best model


```python
def load_and_evaluate(dm, checkpoint_path='./best_model.pt'):
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    # Initialize model
    model = DistilBERTClassifier(
        model_name="distilbert-base-uncased",
        num_classes=dm.num_classes,
        dropout_rate=0.3,
        hidden_dim=256
    ).to(device)
    
    # Load checkpoint
    model.load_best_model(checkpoint_path)
    
    # Load metadata
    metadata_path = os.path.join(os.path.dirname(checkpoint_path), 'training_metadata.pt')
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location=device)
        print(f"  Epoch: {metadata['epoch'] + 1}")
        print(f"  Val Acc: {metadata['val_accuracy']*100:.2f}%")
        print(f"  Test Acc: {metadata['test_accuracy']:.4f}")
    
    # Evaluate
    _, _, test_loader = prepare_data(dm, "distilbert-base-uncased", 48, 32)
    
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print('predicted: ',predicted, 'labels: ',labels)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
    print(f"\n" + "=" * 80)
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 80)
    
    return model, test_acc

# Load and evaluate
best_model, final_acc = load_and_evaluate(dm)
```

    
    ================================================================================
    LOADING MODEL
    ================================================================================
    ✓ Model loaded! Size: 54.85 MB
      Epoch: 5
      Val Acc: 100.00%
      Test Acc: 0.9950



    Map:   0%|          | 0/2000 [00:00<?, ? examples/s]


    Train: 1600 | Val: 200 | Test: 200
    
    Evaluating...


    Testing:  43%|████▎     | 3/7 [00:00<00:00, 23.03it/s]

    predicted:  tensor([3, 4, 3, 5, 4, 3, 2, 3, 2, 2, 5, 5, 1, 4, 1, 1, 4, 1, 2, 1, 5, 3, 5, 2,
            3, 5, 3, 4, 4, 5, 3, 2], device='cuda:0') labels:  tensor([3, 4, 3, 5, 4, 3, 2, 3, 2, 2, 5, 5, 1, 4, 1, 1, 4, 1, 2, 1, 5, 3, 5, 2,
            3, 5, 3, 4, 4, 5, 3, 2], device='cuda:0')
    predicted:  tensor([3, 3, 2, 4, 4, 5, 1, 2, 4, 5, 2, 4, 4, 3, 3, 5, 1, 2, 4, 1, 1, 2, 1, 5,
            3, 2, 2, 5, 1, 3, 3, 1], device='cuda:0') labels:  tensor([3, 3, 2, 4, 4, 5, 1, 2, 4, 5, 2, 4, 4, 3, 3, 5, 1, 2, 4, 1, 1, 2, 1, 5,
            3, 2, 2, 5, 1, 3, 3, 1], device='cuda:0')
    predicted:  tensor([2, 5, 4, 4, 4, 4, 2, 3, 4, 2, 1, 3, 2, 3, 4, 5, 1, 5, 2, 1, 4, 0, 3, 3,
            2, 4, 1, 2, 1, 2, 5, 3], device='cuda:0') labels:  tensor([2, 5, 4, 4, 4, 4, 2, 3, 4, 2, 1, 3, 2, 3, 4, 5, 1, 5, 2, 1, 2, 0, 3, 3,
            2, 4, 1, 2, 1, 2, 5, 3], device='cuda:0')
    predicted:  tensor([2, 2, 4, 4, 1, 3, 1, 1, 1, 2, 1, 2, 2, 3, 3, 1, 5, 1, 3, 4, 1, 1, 3, 4,
            1, 1, 1, 3, 3, 1, 3, 2], device='cuda:0') labels:  tensor([2, 2, 4, 4, 1, 3, 1, 1, 1, 2, 1, 2, 2, 3, 3, 1, 5, 1, 3, 4, 1, 1, 3, 4,
            1, 1, 1, 3, 3, 1, 3, 2], device='cuda:0')
    predicted:  

    Testing: 100%|██████████| 7/7 [00:00<00:00, 23.38it/s]

    tensor([5, 2, 1, 1, 4, 2, 4, 3, 4, 2, 2, 1, 2, 2, 2, 1, 5, 3, 5, 1, 2, 1, 2, 4,
            4, 1, 1, 1, 1, 3, 1, 4], device='cuda:0') labels:  tensor([5, 2, 1, 1, 4, 2, 4, 3, 4, 2, 2, 1, 2, 2, 2, 1, 5, 3, 5, 1, 2, 1, 2, 4,
            4, 1, 1, 1, 1, 3, 1, 4], device='cuda:0')
    predicted:  tensor([5, 1, 2, 5, 1, 2, 2, 3, 1, 1, 2, 4, 1, 5, 2, 2, 2, 2, 4, 5, 2, 3, 2, 1,
            0, 2, 5, 0, 1, 2, 4, 2], device='cuda:0') labels:  tensor([5, 1, 2, 5, 1, 2, 2, 3, 1, 1, 2, 4, 1, 5, 2, 2, 2, 2, 4, 5, 2, 3, 2, 1,
            0, 2, 5, 0, 1, 2, 4, 2], device='cuda:0')
    predicted:  tensor([5, 3, 2, 1, 5, 2, 4, 1], device='cuda:0') labels:  tensor([5, 3, 2, 1, 5, 2, 4, 1], device='cuda:0')
    
    ================================================================================
    Final Test Accuracy: 0.9950 (99.50%)
    ================================================================================


    


##### (i) What is your best model?

Fine-tuned DistilBERT (distilbert-base-uncased) with multi-layer classification head. 

DistilBERT is a knowledge-distilled version of BERT with 40% fewer parameters (66M vs 110M), achieving 97% of BERT's performance while being 60% faster.

##### (ii) The accuracy of your best model on the test set

Best model Test Accuracy: **0.9950**


##### (iii) The values of the hyperparameters of your best model

Hyperparameters:
- Base: distilbert-base-uncased (66M params, 6 layers)
- Learning Rate: 3e-5
- Batch Size: 32
- Dropout: 0.3
- Hidden Dim: 256
- Weight Decay: 0.01
- Warmup: 10%
- Max Length: 48
- Optimizer: AdamW
- Scheduler: Linear with warmup
- Fine-tuning: Last 2 layers + classifier
- Early Stopping: Patience 10

##### (iv) The link to download your best model

best_model.pt

https://drive.google.com/file/d/1DpoSLD-UiuJCPD6tIGnts3CLzkg360CF/view?usp=sharing

training_metadata.pt

https://drive.google.com/file/d/1VvtVN-oVSN4hRFtbE82Y41DdAYucSkj1/view?usp=sharing

---
<div style="text-align: center"> <font color="green">GOOD LUCK WITH YOUR ASSIGNMENT 2!</font> </div>
<div style="text-align: center"> <font color="black">END OF ASSIGNMENT</font> </div>
