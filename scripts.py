def create_word_to_index(token_lists):
    # Initialize the dictionary with special tokens
    word_to_index = {'<SOS>': 0, '<EOS>': 1}
    
    # Start the index from 2 since 0 and 1 are already used
    index = 2
    
    # Iterate through the list of lists
    for token_list in token_lists:
        for token in token_list:
            # Add the token to the dictionary if it's not already present
            if token not in word_to_index:
                word_to_index[token] = index
                index += 1
                
    if '<UNK>' not in word_to_index:
        word_to_index['<UNK>'] = index
        index+=1
    if '<PAD>' not in word_to_index:
        word_to_index['<PAD>'] =index
        index+=1
        
    return word_to_index

def convert_to_indicies(source_list, word_to_index):
  source_indicies = []
  for source in source_list:
    source_indicies.append([word_to_index.get(word,word_to_index['<UNK>']) for word in source])
  return source_indicies


def create_windows(source_list,target_list,window_size):
  source_windows = []
  target_windows = []
  assert len(source_list) == len(target_list)
  for i in range(0,len(source_list),window_size):
    source_window = source_list[i:i+window_size] if i+window_size <= len(source_list) else source_list[i:]
    target_window = target_list[i:i+window_size] if i+window_size <= len(target_list) else target_list[i:]
    source_windows.append(['<SOS>'] + source_window + ['<EOS>'])
    target_windows.append(['<SOS>'] + target_window + ['<EOS>'])
  return source_windows, target_windows


def read_alignments(input_file_name):
  space_token = '<SPACE>'
  source_list = []
  target_list = []

  with open(input_file_name, 'r', encoding='utf-8') as file:
      count = 1  # Initialize count to 0 to correctly skip the header
      for line in file:
          string = line
          line = line.split('\t')
          if count == 1:  # Skip the first line (header)
              count += 1
              continue
          if len(line) == 2:
            if line[1].strip() == '':
              source_list.append(line[0].strip())
              target_list.append(space_token)
            # print(line)
            elif string.startswith('\t'):
              # print(string)
              # print(line)
              # print(line)
              source_list.append(space_token)
              target_list.append(line[1].strip())
            else:
              # print(string)
              # print(line)
              source_list.append(line[0].strip())
              target_list.append(line[1].strip())
          # print(source_list)
          count+=1
          # print(count)

  return source_list, target_list

def read_arabic_for_word2vec(input_filepath):
    all_sentences = []
    arabic_list = []
    with open(input_filepath, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            if line.strip() == ".":
              arabic_list.append(line.strip())
              all_sentences.append(arabic_list)
              arabic_list = []
            else:
              arabic_list.append(line.strip())
    return all_sentences

