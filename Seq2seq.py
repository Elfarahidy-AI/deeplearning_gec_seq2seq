import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class EncoderLSTM(nn.Module):

  def __init__(self,input_size, embedding_dim, embedding_matrix , hidden_size, num_layers, dropout_p=0.1):
    super(EncoderLSTM,self).__init__()
    self.embedding = nn.Embedding(input_size,embedding_dim)
    self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

    self.embedding.weight.requires_grad = False # to prevent trainable embeddings (subject to trial)

    self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout_p) # (number of layers, batch_size, hidden_size)
    self.dropout = SpatialDropout(dropout_p)

  def forward(self,input):

     # input = [max_seq_length,batch size]
    #  print('input to encoder shape ', input.shape)


     embedded = self.dropout(self.embedding(input))
     # embedded = [max_seq_length, batch size, embedding_dim]
    #  print("encoder input to lstm shape", embedded.shape)

     output, (hidden, cell) = self.lstm(embedded)
     return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, output_size,embedding_dim, embedding_matrix, hidden_size, num_layers, dropout_p=0.1):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        # If you want to freeze the embedding layer (no training)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout_p) # hidden size of decoder should be the same as the encoder
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = SpatialDropout(dropout_p)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0) # since input is of shape N (Batch_size)
        # shape of input: (1,Batch_size) --> one word per batch as the decoder outputs one word at a time
        # print("input to decoder shape ",input.shape)
        output = self.dropout(self.embedding(input))
        # shape of output: (1,Batch_size,embedding_dim)
        # print("output of embedding layer shape", output.shape)
        # print("hidden input to decoder shape ", hidden.shape)
        output, (hidden,cell) = self.lstm(output, (hidden,cell))
        # shape of output: (1,Batch_size,hidden_size)
        # print("output of lstm of decoder shape ", output.shape)
        output = self.out(output)
        # print("output of dense of decoder shape ", output.shape)
        # shape of output: (1,Batch_size,output_size(Vocab_size))

        output = output.squeeze(0)
        # print("output of lstm of decoder shape after squeezing ", output.shape)

        # shape of output: (Batch_size,output_size(Vocab_size))
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,source,target,teacher_force_ratio = 0.5):

        batch_size = source.shape[1]  # source shape (seq_len, batch_size)
        # print("batch_size ",batch_size)
        target_len = target.shape[0]    # target shape (seq_len, batch_size)
        # print("target_length ",target_len)
        target_vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(target_len,batch_size,target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        # print("encoder output shape ", hidden.shape)
        # shape of hidden: (num_layers * num_directions, Batch_size, hidden_size)
        # shape of cell: (num_layers * num_directions, Batch_size, hidden_size)
        x = target[0]
        # print("target[0] ",x, x.shape)
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            max_pred = output.argmax(1)
            x = target[t] # if random.random() < teacher_force_ratio else max_pred
        return outputs
    
    def inference(self, source, max_len, start_token_id = 0):
        batch_size = source.shape[1]
        target_vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(max_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)

        # Start token, assuming it's the first token in the target vocab
        x = torch.full((batch_size,), start_token_id, dtype=torch.long).to(device)
        
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            max_pred = output.argmax(1)
            x = max_pred
        
        return outputs


