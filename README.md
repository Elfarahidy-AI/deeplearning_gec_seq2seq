# Overview
This repository contains a Seq2Seq deep learning model designed to perform Arabic grammar and spelling correction. The model is based on an encoder-decoder architecture, where both the encoder and decoder are two-layer LSTM networks with 1042 units each.

# Model Architecture
The Seq2Seq model in this repository is structured as follows:
- Encoder: A two-layer LSTM network with 1042 units in each layer. The encoder processes the input sequence and encodes it into a fixed-size context vector.
- Decoder: A two-layer LSTM network with 1042 units in each layer. The decoder takes the context vector and generates the corrected output sequence.
The architecture is designed to handle the intricacies of Arabic grammar and spelling, ensuring accurate correction and generation of text.

# Dataset
The dataset used for training the model consists of pairs of sentences, where each pair contains a grammatically incorrect or misspelled sentence and its corrected version

# Future Work
- Providing more data to the model
- Adding Attention Mechanism for improving model performace

## License

This project includes code from [Seq2Seq](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq/seq2seq.py), which is licensed under the MIT License. The original code was created by Aladdin Persson.

Additional code and modifications by Hossam Nabil are also licensed under the MIT License.

For more details, see the [LICENSE](License.txt) file.
