import numpy as np
import re
import torch
import utils
from torch.utils.data import TensorDataset, random_split
from transformers import AdamW
import time

# For reproducibility
torch.manual_seed(0)

class NewsQaExample(object):
    '''
    A single training/test set example for News QA.
    
    For examples that do not have any answer, the start and end index should be -1
    '''
    
    def __init__(self, text, question, start_idx = None, end_idx = None, is_training = True):
        '''
        Initializes an object of this class
        
        Parameters
        ------------
        text: str
              The news text
              
        question: str
                  The question text
              
        start_idx: int
                   The character index of starting position of the answer
                   Should be -1 if answer is not present in text
                   Should be None for test examples
        
        end_idx: int
                 The characer index of ending position of the answer
                 Should be -1 if answer is not present in text
                 Should be None for test examples
                   
        is_training: bool
                     If the example is a training example or a test example.
                     Should be True if answer is available, i.e. also for
                     validation examples.
        '''
        # Check for answer indices if input is a training example
        if is_training == True and (start_idx is None or end_idx is None):
            raise AttributeError("Answer start and end indices cannot be `None` for training examples.")
            
        self.doc_text = text
        self.ques_text = question
        self.char_start_idx = start_idx
        self.char_end_idx = end_idx
        self.is_training = is_training
        
        if self.char_start_idx != -1 or self.char_end_idx != -1:
            self.char_to_word_map = self.get_char_to_word_idx()
            if is_training:
                self.word_start_idx = self.char_to_word_map[self.char_start_idx]
                self.word_end_idx = self.char_to_word_map[self.char_end_idx]
                self.is_impossible = False
        
        elif is_training:
            self.is_impossible = True
        
        self.tokens = None
        self.token_to_org_map = None
        self.org_to_token_map = None
        
        self.token_start_idx = None
        self.token_end_idx = None
        self.offset = 0
        
        self.input_ids = None
        self.segment_ids = None
        self.attention_mask = None
    
    def get_char_to_word_idx(self):
        '''
        A functions that returns a list which maps each character index to a word index
        '''
        char_to_word = []
        words = re.split(' ', self.doc_text)

        for idx in range(len(words)):
            # The space next to a word will be considered as part of that word itself
            char_to_word = char_to_word + [idx] * (len(words[idx]) + 1)

        # There is no space after last word, so we need to remove the last element
        char_to_word = char_to_word[:-1]

        # Check for errors
        assert len(char_to_word) == len(self.doc_text)

        return char_to_word
    
    def encode_plus(self, tokenizer, max_seq_len = 512, 
                    max_ques_len = 50, doc_stride = 128, 
                    pad = False):
        '''
        Returns a dictionary with input ids, segment_ids and attention mask
        
        Parameters
        -------------
        tokenizer: obj
                   The tokenizer to use
        
        max_seq_len: int
                     The maximum total sequence length
                     
        max_ques_len: int
                      The maximum length of the question
        
        doc_stride: int
                    For test data, it determines the sliding window stride on text
                    if the news article length is more than max length to produce
                    multiple text-question pairs
                    For training data, it select the window that has the answer
        '''
        self._calculate_input_features(tokenizer, max_seq_len, max_ques_len, 
                                       doc_stride, pad)
        
        # For training example
        if self.is_training:
            features = {"input_ids": torch.tensor([self.input_ids]), 
                        "token_type_ids": torch.tensor([self.segment_ids]), 
                        "attention_mask": torch.tensor([self.attention_mask])}
            
            return features
        
        # For test examples
        else:
            features = list()
            for idx in range(len(self.input_ids)):
                feature = {"input_ids": torch.tensor([self.input_ids[idx]]), 
                           "token_type_ids": torch.tensor([self.segment_ids[idx]]), 
                           "attention_mask": torch.tensor([self.attention_mask[idx]])}
                features.append(feature)
            return features
        
    
    def _calculate_input_features(self, tokenizer, max_seq_len, max_ques_len, 
                                  doc_stride, pad):
        '''
        A function that performs tokenization, calculates token index to original index map, 
        input ids and segment ids
        
        Parameters
        ------------
        Same as encode_plus()
        '''
        ques_tokens = tokenizer.tokenize(self.ques_text)
        
        # Truncate the question upto max_ques_len
        if len(ques_tokens) > max_ques_len:
            ques_tokens = ques_tokens[:max_ques_len]
            
        self.token_to_org_map = []
        self.org_to_token_map = []
        all_doc_tokens = []
        words = re.split(' ', self.doc_text)
        
        for idx, word in enumerate(words):
            self.org_to_token_map.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(word)
            # See if there are sub-tokens for the space-seperated word
            for token in sub_tokens:
                self.token_to_org_map.append(idx)
                all_doc_tokens.append(token)
        
        # Need to update token indices as question comes first
        self.org_to_token_map = np.array(self.org_to_token_map)
        self.org_to_token_map = self.org_to_token_map + min(max_ques_len, len(ques_tokens)) + 2
        
        self.token_to_org_map = [-1] * (min(max_ques_len, len(ques_tokens)) + 2) + self.token_to_org_map + [-1]
        self.token_to_org_map = np.array(self.token_to_org_map)
        
        # -3 is for [CLS], [SEP], [SEP] tokens
        max_doc_len = max_seq_len - min(max_ques_len, len(ques_tokens)) - 3
        
        if self.is_training and self.is_impossible:
            # Set to [CLS] token index
            self.token_start_idx = 0
            self.token_end_idx = 0
        
        # For Train/Validation data
        if self.is_training:
            
            if not self.is_impossible:
                # Get token indices from word indices
                self.token_start_idx = self.org_to_token_map[self.word_start_idx]
                self.token_end_idx = self.org_to_token_map[self.word_end_idx]
            
            # If the answer ends before max_doc_len
            if self.token_end_idx < max_doc_len:
                # No need to change answer indices
                self.offset = 0
                # Truncate doc tokens
                doc_tokens = all_doc_tokens[:max_doc_len]
                
            else:
                # Use a slinding window to find the window that has the 
                # answer in it
                for i in range(0, len(all_doc_tokens), doc_stride):
                    # Check if answer is in that window
                    if self.token_start_idx >= i and self.token_end_idx < i + max_doc_len:
                        # Need to chane answer indices by subtracting i
                        self.offset = i
                        # Select that window as document token
                        doc_tokens = all_doc_tokens[i:i + max_doc_len]
                    else:
                        continue
            
            # Update the indices based on offset
            self.token_start_idx = self.token_start_idx - self.offset
            self.token_end_idx = self.token_end_idx - self.offset
            
            # The final training tokens
            self.tokens = ['[CLS]'] + ques_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
            
            # Input ids
            self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
            
            # Segment ids
            self.segment_ids = [0] * (len(ques_tokens) + 2) + [1] * (len(doc_tokens) + 1)
            
            # Attention mask
            self.attention_mask = [1] * len(self.input_ids)
            
            # Check for errors
            assert len(self.tokens) == len(self.input_ids)
            assert len(self.input_ids) == len(self.segment_ids)
            assert len(self.attention_mask) == len(self.segment_ids)
            
            if pad:
                # Add padding
                self.input_ids = self.add_padding(self.input_ids, max_seq_len, 
                                                  tokenizer.pad_token_id)
                self.segment_ids = self.add_padding(self.segment_ids, 
                                                    max_seq_len, 1)
                self.attention_mask = self.add_padding(self.attention_mask, 
                                                       max_seq_len, 0)
                
                # Check for errors
                assert len(self.input_ids) == max_seq_len
                assert len(self.segment_ids) == max_seq_len
                assert len(self.attention_mask) == max_seq_len
        
        # Test data
        else:
            self.tokens = []
            self.input_ids = []
            self.segment_ids = []
            self.attention_mask = []
            
            # If the length of tokens is less than max_doc_len, just use a single question-text pair
            if len(all_doc_tokens) < max_doc_len:
                input_tokens = ['[CLS]'] + ques_tokens + ['[SEP]'] + all_doc_tokens + ['[SEP]']
                input_id = tokenizer.convert_tokens_to_ids(input_tokens)
                segment_id = [0] * (len(ques_tokens) + 2) + [1] * (len(all_doc_tokens) + 1)
                attn_msk = [1] * len(input_id)
                
                # Check for errors
                assert len(input_tokens) == len(input_id)
                assert len(input_id) == len(segment_id)
                assert len(segment_id) == len(attn_msk)
                
                if pad:
                    # Add padding
                    input_id = self.add_padding(input_id, max_seq_len, 
                                                tokenizer.pad_token_id)
                    segment_id = self.add_padding(segment_id, 
                                                  max_seq_len, 1)
                    attn_msk = self.add_padding(attn_msk, 
                                                max_seq_len, 0)
                    
                    # Check for errors
                    assert len(input_id) == max_seq_len
                    assert len(segment_id) == max_seq_len
                    assert len(attn_msk) == max_seq_len
                
                self.tokens.append(input_tokens)
                self.input_ids.append(input_id)
                self.segment_ids.append(segment_id)
                self.attention_mask.append(attn_msk)

            # If not, create multiple question-text pairs using a sliding window on text
            else:
                for i in range(0, len(all_doc_tokens), doc_stride):
                    doc_tokens = all_doc_tokens[i:i+max_doc_len]
                    input_tokens = ['[CLS]'] + ques_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
                    input_id = tokenizer.convert_tokens_to_ids(input_tokens)
                    segment_id = [0] * (len(ques_tokens) + 2) + [1] * (len(doc_tokens) + 1)
                    attn_msk = [1] * len(input_id)
                    
                    # Check for errors
                    assert len(input_tokens) == len(input_id)
                    assert len(input_id) == len(segment_id)
                    assert len(segment_id) == len(attn_msk)
                    
                    if pad:
                        # Add padding
                        input_id = self.add_padding(input_id, max_seq_len, 
                                                    tokenizer.pad_token_id)
                        segment_id = self.add_padding(segment_id, 
                                                      max_seq_len, 1)
                        attn_msk = self.add_padding(attn_msk, 
                                                    max_seq_len, 0)
                        
                        # Check for errors
                        assert len(input_id) == max_seq_len
                        assert len(segment_id) == max_seq_len
                        assert len(attn_msk) == max_seq_len
                    
                    self.tokens.append(input_tokens)
                    self.input_ids.append(input_id)
                    self.segment_ids.append(segment_id)
                    self.attention_mask.append(attn_msk)
        
        # Determine that features are calculated
        self.features_calculated = True
                        
    def get_ans_char_range(self, token_start_idx, token_end_idx, offset = None):
        '''
        A function that returns the character start and end index from
        respective token indices
        
        Parameters
        ------------
        token_start_idx: int
                         The start index referring to tokens
        
        token_end_idx: int
                       The end index referring to tokens
        '''
        if offset is None:
            offset = self.offset
        
        try:   
            # Update indices based on offset
            token_start_idx = token_start_idx + offset
            token_end_idx = token_end_idx + offset

            # Getting word indices from token indices
            start_word_idx = self.token_to_org_map[token_start_idx]
            end_word_idx = self.token_to_org_map[token_end_idx]

            # If the indices are a part of the question, it means that there
            # is no answer
            if start_word_idx == -1 or end_word_idx == -1:
                return (0, 0)

            # Getting char indices from word indices
            start_char_idx = self.char_to_word_map.index(start_word_idx)
            end_char_idx = self.char_to_word_map.index(end_word_idx + 1)
        
        except:
            start_char_idx = 0
            end_char_idx = 0

        return (start_char_idx, end_char_idx)
    
    def add_padding(self, seq, max_len, pad_value):
        '''
        A function that adds padding to sequences
        
        Parameters
        ------------
        seq: list
             The unpadded sequence
             
        max_len: int
                 The length of final sequence
                 
        pad_value: int
                   The value to pad
        '''
        return seq + [pad_value] * (max_len - len(seq))
    
    def get_label(self):
        '''
        A function that returns the start and end token index of the answer
        '''
        return [self.token_start_idx, self.token_end_idx]
    
    def __repr__(self):
        '''
        Representation of an object of this class
        '''
        string = ""
        string += "text: " + self.doc_text
        string += "\n\nquestion: " + self.ques_text
        if self.char_start_idx is not None:
            string += "\n\nanswer: " + self.doc_text[self.char_start_idx:self.char_end_idx]
        else:
            string += "\n\nanswer: n/a"
        
        return string
    
    def __str__(self):
        return self.__repr__()
  

class NewsQaModel(object):
    '''
    A question answering model
    '''
    def __init__(self, model = None):
        '''
        Initializes an object of this class
        '''
        self.model = model
    
    def train(self, training_set, eval_set, feature_idx_map, device, 
              optimizer = None, num_epochs = 3, lr = 1e-5, filename = 'model.pt'):
        '''
        Trains the model
        
        Parameters
        ------------
        training_set: torch.utils.data.DataLoader
                      The training set divided into batches
        
        eval_set: torch.utils.data.DataLoader
                  The evaluation set divided into batches
                  
        feature_idx_map: dict
                         Mappings of input features to element 
                         indices in a batch
        
        device: torch.device
                The device (CPU/GPU) to use for training
        
        optimizer: The optimizer to use
        
        num_epochs: int
                    The number of epochs to train the model for
                    
        lr: float
            The learning rate to use
            
        filename: str
                  The name of the file to save the best model in
        '''
        self.model.to(device)
        
        if optimizer is None:
            optimizer = AdamW(self.parameters(), lr = lr)
            
        self.train_losses = []
        self.train_f1_scores = []
        self.train_acc = []
        
        self.val_losses = []
        self.val_f1_scores = []
        self.val_acc = []
        
        # Keeping track of maximum accuracy to save the model
        max_acc = 0
        
        # Length of training set
        train_len = len(training_set)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print("Epoch {}/{}:".format(epoch + 1, num_epochs))
            # Put the model in training mode
            self.model.train()
            
            total_loss = 0
            total_f1 = 0
            total_acc = 0
            
            for idx, batch in enumerate(training_set):
                # Sending the data to GPU
                batch = [b.to(device) for b in batch]
                input_features = {feature: batch[i] for feature, i in feature_idx_map.items()}
                
                # Clear previous gradients
                self.model.zero_grad()
                
                # Calculate the outputs
                outputs = self.model(**input_features)
                loss, start_scores, end_scores = outputs
                metric = self.calculate_metrics(start_scores, end_scores, 
                                                input_features['start_positions'], 
                                                input_features['end_positions'])
                
                
                total_f1 += metric['f1']
                total_acc += metric['acc']
                total_loss += loss.item()
                
                # Backpropogate
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Print progress
                if idx != len(training_set) - 1:
                    string = '\tloss: {:.4f}\tf1: {:.4f}\tacc: {:.4f}'
                    string = string.format(loss.item(), metric['f1'], metric['acc'])
                    utils.drawProgressBar(idx + 1, train_len, string)
                
                # Empty torch cache to free memory
                torch.cuda.empty_cache()
            
            # The average metrics over the epoch
            avg_loss = total_loss/train_len
            avg_f1 = total_f1/train_len
            avg_acc = total_acc/train_len
            
            # Calculate validation loss
            val_metrics = self.evaluate(eval_set, feature_idx_map, device, verbose = 0)
            
            # Run time of epoch
            end_time = time.time()
            runtime = end_time - start_time
            time_min = int(np.floor(runtime/60))
            time_sec = int(runtime%60)
            
            # Print validation metrics at the end of epoch
            string = '  {}m {}s'.format(time_min, time_sec)
            string += '\tloss: {:.4f}\tf1: {:.4f}\tacc: {:.4f}'
            string = string.format(avg_loss, avg_f1, avg_acc)
            string += '\tval_loss: {:.4f}\tval_f1: {:.4f}\tval_acc: {:.4f}'
            string = string.format(val_metrics['loss'], val_metrics['f1'], val_metrics['acc'])
            utils.drawProgressBar(train_len, train_len, string)
            
            # Save best model
            if val_metrics['acc'] > max_acc:
                string = '\nValidation accuracy increased from {:.4f} to {:.4f}, saving to {}'
                string = string.format(max_acc, val_metrics['acc'], filename)
                print(string)
                max_acc = val_metrics['acc']
                self.save(filename)
                
            if epoch != num_epochs - 1:
                print('\n\n')
            
            self.train_losses.append(avg_loss)
            self.train_f1_scores.append(avg_f1)
            self.train_acc.append(avg_acc)
            
            self.val_f1_scores.append(val_metrics['f1'])
            self.val_losses.append(val_metrics['loss'])
            self.val_acc.append(val_metrics['acc'])
        
        # Put the model back into evaluation mode
        self.model.eval()
    
    
    def evaluate(self, eval_set, feature_idx_map, device, verbose = 1):
        '''
        Evaluates the model
        
        Parameters
        -----------
        eval_set: torch.utils.data.DataLoader
                  Evaluation dataset
                  
        feature_idx_map: dict
                         Mappings of input features to element 
                         indices in a batch
                         
        device: torch.device
                The device to use for evaluation
        '''
        # Set the model in evaluation mode
        self.model.eval()
        total_loss = 0
        total_f1 = 0
        total_acc = 0
        eval_len = len(eval_set)
        
        for idx, batch in enumerate(eval_set):
            batch = [b.to(device) for b in batch]
            
            # Deactivate autograd
            with torch.no_grad():
                # Get features as a dictionary of input parameters
                input_features = {feature: batch[i] for feature, i in feature_idx_map.items()}
                # Get the output
                outputs = self.model(**input_features)
                
                loss, start_scores, end_scores = outputs
                metric = self.calculate_metrics(start_scores, end_scores, 
                                                 input_features['start_positions'], 
                                                 input_features['end_positions'])
                
                total_loss += loss.item()
                total_f1 += metric['f1']
                total_acc += metric['acc']
                if verbose == 1:
                    utils.drawProgressBar(idx + 1, eval_len)
                
        metrics = {'loss': total_loss/eval_len, 'f1': total_f1/eval_len, 
                   'acc': total_acc/eval_len}
        if verbose == 1:
            print("\nloss: {:.4f}\tf1:{:.4f}\tacc:{:.4f}".format(metrics['loss'], metrics['f1'], metrics['acc']))
        
        return metrics
    
    
    def calculate_metrics(self, start_scores, end_scores, start_idx, end_idx):
        '''
        Calculates the f1 score for a batch of examples
        
        Parameters
        -----------
        start_scores, end_scores: torch.tensor
                                  The index scores predicted by model
        
        start_idx, end_idx: torch.tensor()
                            The actual indices
        '''
        # Convert variables to numpy arrays
        start_scores = start_scores.cpu().detach().numpy()
        end_scores = end_scores.cpu().detach().numpy()
        start_idx = start_idx.cpu().numpy()
        end_idx = end_idx.cpu().numpy() + 1
        
        # Get the predicted indices from scores
        pred_start = np.argmax(start_scores, axis = 1)
        pred_end = np.argmax(end_scores, axis = 1) + 1
        
        f1_scores = []
        correct = 0
        
        for idx in range(len(start_idx)):
            overlap = set(range(start_idx[idx], end_idx[idx])).intersection(range(pred_start[idx], pred_end[idx]))
            overlap = len(overlap)
            
            # If either of them have no answer
            if end_idx[idx] == 0 or pred_end[idx] == 0:
                f1_scores.append(int(end_idx[idx] == pred_end[idx]))
                correct += int(end_idx[idx] == pred_end[idx])
                continue
            # If they don't overlap at all
            if overlap == 0 or pred_start[idx] >= pred_end[idx]:
                f1_scores.append(0)
                correct += 0
                continue
            
            # If there is an overlap, we consider it correct
            correct += 1
            
            precision = overlap / (pred_end[idx] - pred_start[idx])
            recall = overlap / (end_idx[idx] - start_idx[idx])
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        
        metrics = {'f1': sum(f1_scores)/len(f1_scores), 'acc': correct/len(f1_scores)}
        return metrics
    
    def save(self, filename):
        '''
        Saves the model
        '''
        torch.save(self.model, filename)
        
    def load(self, filename):
        '''
        Loads a saved model
        '''
        self.model = torch.load(filename, map_location=torch.device('cpu'))
        
    def parameters(self):
        '''
        Returns model parameters
        '''
        return self.model.parameters()
    
    def predict(self, **input_features):
        '''
        Returns the outputs of the model
        '''
        self.model.to(torch.device("cpu"))
        return self.model(**input_features)
    
    def __repr__(self):
        return self.model.__repr__()
    
    def __str__(self):
        return self.model.__str__()


def create_dataset(features, labels, model = "bert", 
                   train_ratio = 0.7, val_ratio = 0.1):
    '''
    A function that creates train, validation and test 
    dataset of type TensorDataset
    Also returns a dictionary that maps feature name to index
    
    Parameters
    ------------
    features: list
              A list of dictionaries that has all the features
              All features should be of same size
    
    labels: list or tuple
            A list or tuple of length 2, where the first position
            is the start index and second position is end index 
            of the answer
    
    model: str
           The type of model to be used on the dataset
           For example bert, xlm, xlnet, roberta
    
    train_ratio: float
                 The ratio of train dataset, between 0 and 1
    
    val_ratio: float
               The ratio of validation dataset, between 0 and 1
    '''
    input_ids = []
    segment_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []
    
    # XLNET also has is_impossible input feature
    # that determines if there is no answer present
    is_impossible = []
    
    total_examples = len(features)
    for idx in range(total_examples):
        f = features[idx]
        input_ids.append(f['input_ids'])
        segment_ids.append(f['token_type_ids'])
        attention_masks.append(f['attention_mask'])
        start_positions.append(labels[idx][0])
        end_positions.append(labels[idx][1])
        
        is_impossible.append(1 if labels[idx][0] == 0 else 0)
    
    # Converting lists to tensors
    input_ids = torch.cat(input_ids, dim = 0)
    segment_ids = torch.cat(segment_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    start_positions = torch.tensor(start_positions)
    end_positions = torch.tensor(end_positions)
    is_impossible = torch.tensor(is_impossible)
    
    # Creating a TensorDataset
    if model == "xlnet":
        dataset = TensorDataset(input_ids, segment_ids, attention_masks, 
                                is_impossible, start_positions, end_positions)
        # Dictionary that maps each feature name to its index in the dataset
        feature_idx_map = {'input_ids': 0, 'token_type_ids': 1, 
                           'attention_mask': 2, 'is_impossible': 3, 
                           'start_positions': 4, 'end_positions': 5}
    elif model == "distilbert":
        dataset = TensorDataset(input_ids, attention_masks, 
                                start_positions, end_positions)
        feature_idx_map = {'input_ids': 0, 'attention_mask': 1, 
                           'start_positions': 2, 'end_positions': 3}
    else:
        dataset = TensorDataset(input_ids, segment_ids, attention_masks, 
                                start_positions, end_positions)
        feature_idx_map = {'input_ids': 0, 'token_type_ids': 1, 
                           'attention_mask': 2, 'start_positions': 3, 
                           'end_positions': 4}
    
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - val_size - train_size
    
    # Splitting the dataset into train, validation and test
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    return train_set, val_set, test_set, feature_idx_map


def get_single_prediction(text, question, tokenizer, newsqa_model, 
                          max_seq_len = 512, max_ques_len = 50, 
                          doc_stride = 128, pad = True):
    '''
    Returns the model prediction for a single example
    
    Parameters
    ------------
    text: str
          The news text to use
    
    question: str
              The question to answer
    
    newsqa_model: NewsQaModel object
                  The model to use for prediction
    '''
    # Create an example object
    ex = NewsQaExample(text, question, is_training=False)
    # Get list of features, by dividing text into multiple parts by stride
    features = ex.encode_plus(tokenizer, max_seq_len, max_ques_len, doc_stride, pad)
    outputs = []
    ans_texts = []
    
    for idx, input_feature in enumerate(features):
        # Get predictions
        start_scores, end_scores = newsqa_model.predict(**input_feature)
        
        start_scores = start_scores.cpu().detach().numpy()
        end_scores = end_scores.cpu().detach().numpy()
        
        start_idx = int(np.argmax(start_scores))
        end_idx = int(np.argmax(end_scores))
        
        # Get the character indices from token indices
        char_start_idx, char_end_idx = ex.get_ans_char_range(start_idx, end_idx, 
                                                             offset = idx * doc_stride)
        outputs.append((char_start_idx, char_end_idx))
        ans_texts.append(text[char_start_idx:char_end_idx])
    
    return ans_texts, outputs
        