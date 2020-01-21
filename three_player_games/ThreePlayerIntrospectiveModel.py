import json
import logging
import os
import random
import sys
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from transformers import BertModel, BertTokenizer, BertForSequenceClassification

# classes needed for Rationale3Player
class RnnModel(nn.Module):
    def __init__(self, args, input_dim):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.layer_num -- number of RNN layers   
        args.cell_type -- type of RNN cells, GRU or LSTM
        """
        super(RnnModel, self).__init__()
        
        self.args = args
 
        if args.cell_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=input_dim, 
                                    hidden_size=args.hidden_dim//2, 
                                    num_layers=args.layer_num, bidirectional=True, dropout=.5)
        elif args.cell_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=input_dim, 
                                     hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True, dropout=.5)
    
    def forward(self, embeddings, mask=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            mask -- a float tensor of masks, (batch_size, length)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)
        """
        embeddings_ = embeddings.transpose(0, 1) #(sequence_length, batch_size, embedding_dim)
        
        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = list(map(int, seq_lengths))
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embeddings_, seq_lengths)
        else:
            inputs_ = embeddings_
        
        hidden, _ = self.rnn_layer(inputs_) #(sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
        
        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden) #(length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0) #(batch_size, hidden_dim, sequence_length)

class ClassifierModule(nn.Module):
    '''
    classifier for both E and E_anti models provided with RNP paper code
    '''
    def __init__(self, args):
        super(ClassifierModule, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        self.input_dim = args.embedding_dim
        
        self.encoder = RnnModel(self.args, self.input_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.NEG_INF = -1.0e6
        

    def forward(self, word_embeddings, z, mask):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """        

        masked_input = word_embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, mask)
        
        max_hidden = torch.max(hiddens + (1 - mask * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
        predict = self.predictor(max_hidden)
        return predict


# (extra) classes needed for HardRationale
class CnnModel(nn.Module):
    def __init__(self, args):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.kernel_size -- kernel size of the conv1d
        args.layer_num -- number of CNN layers
        """
        super(CnnModel, self).__init__()

        self.args = args
        if args.kernel_size % 2 == 0:
            raise ValueError("args.kernel_size should be an odd number")
            
        self.conv_layers = nn.Sequential()
        for i in range(args.layer_num):
            if i == 0:
                input_dim = args.embedding_dim
            else:
                input_dim = args.hidden_dim
            self.conv_layers.add_module('conv_layer{:d}'.format(i), nn.Conv1d(in_channels=input_dim, 
                                                  out_channels=args.hidden_dim, kernel_size=args.kernel_size,
                                                                             padding=(args.kernel_size-1)/2))
            self.conv_layers.add_module('relu{:d}'.format(i), nn.ReLU())
        
    def forward(self, embeddings):
        """
        Given input embeddings in shape of (batch_size, sequence_length, embedding_dim) generate a 
        sentence embedding tensor (batch_size, sequence_length, hidden_dim)
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)       
        """
        embeddings_ = embeddings.transpose(1, 2) #(batch_size, embedding_dim, sequence_length)
        hiddens = self.conv_layers(embeddings_)
        return hiddens
    
class DepRnnModel(nn.Module):

    def __init__(self, args, input_dim):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.layer_num -- number of RNN layers   
        args.cell_type -- type of RNN cells, GRU or LSTM
        """
        super(DepRnnModel, self).__init__()
        
        self.args = args
 
        if args.cell_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=input_dim, 
                                    hidden_size=args.hidden_dim//2, 
                                    num_layers=args.layer_num, bidirectional=True)
        elif args.cell_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=input_dim, 
                                     hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True)
    
    def forward(self, embeddings, h0=None, c0=None, mask=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            mask -- a float tensor of masks, (batch_size, length)
            h0, c0 --  (num_layers * num_directions, batch, hidden_size)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)
        """
        embeddings_ = embeddings.transpose(0, 1) #(sequence_length, batch_size, embedding_dim)
        
        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = list(map(int, seq_lengths))
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embeddings_, seq_lengths)
        else:
            inputs_ = embeddings_
        
        if self.args.cell_type == 'GRU' and h0 is not None:
            hidden, _ = self.rnn_layer(inputs_, h0)
        elif self.args.cell_type == 'LSTM' and h0 is not None and c0 is not None:
            hidden, _ = self.rnn_layer(inputs_, (h0, c0)) #(sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
        else:
            hidden, _ = self.rnn_layer(inputs_)
        
        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden) #(length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0) #(batch_size, hidden_dim, sequence_length)

class DepGenerator(nn.Module):
    
    def __init__(self, args, input_dim):
        """        
        args.z_dim -- rationale or not, always 2
        args.model_type -- "CNN" or "RNN"

        if CNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.kernel_size -- kernel size of the conv1d
            args.layer_num -- number of CNN layers        
        if use RNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.layer_num -- number of RNN layers   
            args.cell_type -- type of RNN cells, "GRU" or "LSTM"
        """
        super(DepGenerator, self).__init__()
        
        self.args = args
        self.z_dim = args.z_dim
        
        self.rnn_model = DepRnnModel(args, input_dim)
        self.output_layer = nn.Linear(args.hidden_dim, self.z_dim)
        
        
    def forward(self, x, h0=None, c0=None, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        
        #(batch_size, sequence_length, hidden_dim)
        hiddens = self.rnn_model(x, h0, c0, mask).transpose(1, 2).contiguous() 
        scores = self.output_layer(hiddens) # (batch_size, sequence_length, 2)

        return scores
        
class IntrospectionGeneratorModule(nn.Module):
    '''
    classifier for both E and E_anti models
    '''
    def __init__(self, args):
        super(IntrospectionGeneratorModule, self).__init__()
        self.args = args
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        self.label_embedding_dim = args.label_embedding_dim
        
        self.fixed_classifier = args.fixed_classifier
        
        self.input_dim = args.embedding_dim
        self.NEG_INF = -1.0e6
        self.lab_embed_layer = self._create_label_embed_layer() # should be shared with the Classifier_pred weights
        
        # baseline classification model
        # self.Classifier_enc = RnnModel(args, self.input_dim)
        # self.Classifier_pred = nn.Linear(self.hidden_dim, self.num_labels)

        self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2, output_hidden_states=True, output_attentions=True)
        # self.Classifier_pred = nn.Linear(self.hidden_dim, self.num_labels)

        self.Transformation = nn.Sequential()
        self.Transformation.add_module('linear_layer', nn.Linear(self.hidden_dim + self.label_embedding_dim, self.hidden_dim // 2))
        self.Transformation.add_module('tanh_layer', nn.Tanh())
        self.Generator = DepGenerator(args, self.input_dim)
        
    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.label_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer
    
    def forward(self, X_tokens, mask):
        # cls_hiddens = self.Classifier_enc(word_embeddings, mask) # (batch_size, hidden_dim, sequence_length)
        # max_cls_hidden = torch.max(cls_hiddens + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        
        # if self.fixed_classifier:
            # max_cls_hidden = Variable(max_cls_hidden.data)
        
        # cls_pred_logits = self.Classifier_pred(max_cls_hidden) # (batch_size, num_labels)
        cls_pred_logits, hidden_states, _ = self.classifier(X_tokens, attention_mask=mask)

        # print("mask: ", mask.shape, mask.unsqueeze(1).shape)
        # print("hiddenstates:", hidden_states[-1].shape)
        max_cls_hidden = torch.max(hidden_states[-1].transpose(1, 2) + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        # print("max cls hidden shape:", max_cls_hidden.shape)
        if self.fixed_classifier:
            max_cls_hidden = Variable(max_cls_hidden.data)

        word_embeddings = hidden_states[0]

        # _, cls_pred = torch.max(cls_pred_logits, dim=1) # (batch_size,)
        _, cls_pred = torch.max(cls_pred_logits, dim=1)

        cls_lab_embeddings = self.lab_embed_layer(cls_pred) # (batch_size, lab_emb_dim)
        
        init_h0 = self.Transformation(torch.cat([max_cls_hidden, cls_lab_embeddings], dim=1)) # (batch_size, hidden_dim / 2)
        init_h0 = init_h0.unsqueeze(0).expand(2, init_h0.size(0), init_h0.size(1)).contiguous() # (2, batch_size, hidden_dim / 2)
        
        z_scores_ = self.Generator(word_embeddings, h0=init_h0, mask=mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF
        
        return z_scores_, cls_pred_logits, word_embeddings


class BertPreprocessor():

    def __init__(self, tokenizer = None, model = None, max_length = 50, pad_to_max = True):

        # super(BertPreprocessor, self).__init__(config)
        # self.num_labels = config.num_labels

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # if model:
        #     self.model = model
        # else:
        #     self.model = BertModel.from_pretrained('bert-base-uncased')
        #     self.model.cuda()

        self.max_length = max_length
        self.pad_to_max = True

        # self.init_weights()

    def encode(self, X_text):
        '''
        turn text to tokens
        X_text: batch_size * 1
        '''
        input_ids = []
        attention_mask = []
        counts = []
        for text in X_text:
            d = self.tokenizer.encode_plus(text, max_length = self.max_length, pad_to_max_length=self.pad_to_max)
            input_ids.append(d['input_ids'])
            attention_mask.append(d['attention_mask'])
            counts.append(sum(d['attention_mask']))
        
        return input_ids, attention_mask, counts

    def decode_single(self, id_list):
        '''
        id_list: a list of token ids
        '''
        return self.tokenizer.convert_ids_to_tokens(id_list)
        


    # def embed(self, X_tokens, X_mask):
    #     '''
    #     turn tokens to embeddings
    #     '''
    #     #TODO: do we want with torch.no_grad():
    #     #TODO: cuda?
    #     embeddings = self.model(X_tokens, X_mask)[0]
    #     return embeddings



class ThreePlayerModel(nn.Module):
    """flattening the HardIntrospectionRationale3PlayerClassificationModel -> HardRationale3PlayerClassificationModel -> 
       Rationale3PlayerClassificationModel dependency structure from original paper code"""

    def __init__(self, args, preprocessor, num_labels, explainer=ClassifierModule, anti_explainer=ClassifierModule, generator=IntrospectionGeneratorModule, classifier=ClassifierModule):
        """Initializes the model, including the explainer, anti-rationale explainer
        Args:
            args: 
            embeddings:
            need: num_labels/size of label vocab
            classificationModule: type of classifier
            
        """
        super(ThreePlayerModel, self).__init__()
        self.args = args
        # attributes that would need to be ported over from args
        # from Rationale3PlayerClassificationModel initialization
        self.model_type = args.model_type
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        self.input_dim = args.embedding_dim
        # from Hardrationale3PlayerClassificationModel initialization
        self.highlight_percentage = args.highlight_percentage
        self.highlight_count = args.highlight_count
        self.exploration_rate = args.exploration_rate
        self.margin = args.margin
        # from HardRationale3PlayerClassificationModelForEmnlp initialization:
        self.game_mode = args.game_mode
        self.ngram = args.ngram
        self.lambda_acc_gap = args.lambda_acc_gap
        self.z_history_rewards = deque([0], maxlen=200)

        # initialize model components
        self.E_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2, output_hidden_states=False, output_attentions=False)
        self.E_anti_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2, output_hidden_states=False, output_attentions=False)
        # self.C_model = classifier(args)
        self.generator = generator(args)
                    
        # Independent inputs: embeddings, num_labels
        # self.vocab_size, self.embedding_dim = embeddings.shape
        # self.embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.embed_layer.weight.data = torch.from_numpy(embeddings)
        # self.embed_layer.weight.requires_grad = self.args.fine_tuning
        self.num_labels = num_labels 
  
        # no internal code dependencies
        self.NEG_INF = -1.0e6 # TODO: move out and set as constant?
        self.loss_func = nn.CrossEntropyLoss(reduce=False)

        # what are these for?
        self.count_tokens = args.count_tokens
        self.count_pieces = args.count_pieces

        self.fixed_classifier = args.fixed_classifier

        self.preprocessor = preprocessor # should be something like bert model, run encoder.embed(X_text)

    # methods from Hardrationale3PlayerClassificationModel
    def init_optimizers(self): # not sure if this can be merged with initializer
        self.freeze_bert_classifier(self.E_model, entire=False)
        self.freeze_bert_classifier(self.E_anti_model, entire=False)

        self.opt_E = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_model.parameters()), lr=self.args.lr)
        self.opt_E_anti = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_anti_model.parameters()), lr=self.args.lr)
    
    def init_rl_optimizers(self):
        print("num gen classifier params:", len(list(self.generator.classifier.parameters())))
        print("num gen params:", len(list(self.generator.parameters())))
        self.freeze_bert_classifier(self.generator.classifier, entire=False)
        # for name, param in self.generator.classifier.named_parameters():
        #     if "bert.embeddings" in name or ("bert.encoder" in name and "layer.11" not in name):
        #         param.requires_grad = False
        #     print(name, param.requires_grad)
        
        self.opt_G_sup = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.classifier.parameters())) #TODO twiddle around with learning rate?
        self.opt_G_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.args.lr * 0.1)
    
    def freeze_bert_classifier(self, classifier, entire=False):
        if entire:
            for name, param in classifier.named_parameters():
                param.requires_grad = False
        else:
            for name, param in classifier.named_parameters():
                if "bert.embeddings" in name or ("bert.encoder" in name and "layer.11" not in name):
                    param.requires_grad = False

    def _generate_rationales(self, z_prob_):
        '''
        Input:
            z_prob_ -- (num_rows, length, 2)
        Output:
            z -- (num_rows, length)
        '''        
        z_prob__ = z_prob_.view(-1, 2) # (num_rows * length, 2)
        
        # sample actions
        sampler = torch.distributions.Categorical(z_prob__)
        if self.training:
            z_ = sampler.sample() # (num_rows * p_length,)
        else:
            z_ = torch.max(z_prob__, dim=-1)[1]
        
        #(num_rows, length)
        z = z_.view(z_prob_.size(0), z_prob_.size(1))
        
        if self.args.cuda:
            z = z.type(torch.cuda.FloatTensor)
        else:
            z = z.type(torch.FloatTensor)
            
        # (num_rows * length,)
        neg_log_probs_ = -sampler.log_prob(z_)
        # (num_rows, length)
        neg_log_probs = neg_log_probs_.view(z_prob_.size(0), z_prob_.size(1))
        
        return z, neg_log_probs
        
    # methods from emnlp model
    def count_regularization_baos_for_both(self, z, count_tokens, count_pieces, mask=None):
        """
        Compute regularization loss, based on a given rationale sequence
        Use Yujia's formulation

        Inputs:
            z -- torch variable, "binary" rationale, (batch_size, sequence_length)
            percentage -- the percentage of words to keep
        Outputs:
            a loss value that contains two parts:
            continuity_loss --  \sum_{i} | z_{i-1} - z_{i} | 
            sparsity_loss -- |mean(z_{i}) - percent|
        """

        # (batch_size,)
        if mask is not None:
            mask_z = z * mask
            seq_lengths = torch.sum(mask, dim=1)
        else:
            mask_z = z
            seq_lengths = torch.sum(z - z + 1.0, dim=1)
        
        mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
            
        continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
        percentage = count_pieces * 2 / seq_lengths
        continuity_loss = torch.abs(continuity_ratio - percentage)
        
        sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
        percentage = count_tokens / seq_lengths #(batch_size,)
        sparsity_loss = torch.abs(sparsity_ratio - percentage)

        return continuity_loss, sparsity_loss

    def train_one_step(self, X_tokens, label, baseline, mask):
        # TODO: try to see whether removing the follows makes any differences
        self.opt_E_anti.zero_grad()
        self.opt_E.zero_grad()
        self.opt_G_sup.zero_grad()
        self.opt_G_rl.zero_grad()
        # self.generator.classifier.zero_grad()
        
        predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(X_tokens, mask)
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
#         e_loss = torch.mean(self.loss_func(predict, label))
        _, cls_pred = torch.max(cls_predict, dim=1) # (batch_size,)
#         e_loss = torch.mean(self.loss_func(predict, cls_pred)) # e_loss comes from only consistency
        e_loss = (torch.mean(self.loss_func(predict, label)) + torch.mean(self.loss_func(predict, cls_pred))) / 2
        
        # g_sup_loss comes from only cls pred loss
        g_sup_loss, g_rl_loss, rewards, consistency_loss, continuity_loss, sparsity_loss = self.get_loss(predict, 
                                                                         anti_predict, 
                                                                         cls_predict, label, z, 
                                                                         neg_log_probs, baseline, mask)
        
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_sup_loss':g_sup_loss.cpu().data, 'g_rl_loss':g_rl_loss.cpu().data}
        
        e_loss_anti.backward(retain_graph=True)
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward(retain_graph=True)
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        if not self.fixed_classifier:
            g_sup_loss.backward(retain_graph=True)
            self.opt_G_sup.step()
            self.opt_G_sup.zero_grad()

        g_rl_loss.backward(retain_graph=True)
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        return losses, predict, anti_predict, cls_predict, z, rewards, consistency_loss, continuity_loss, sparsity_loss
        
    def forward(self, X_tokens, X_mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """
        # X_tokens, X_mask = self.preprocessor.encode(X_text)
        # word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        # word_embeddings = self.preprocessor.embed(X_tokens, X_mask)

        z_scores_, cls_predict, word_embeddings = self.generator(X_tokens, X_mask)
        
        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (X_mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - X_mask.unsqueeze(-1)) * z_probs_)
        
        z, neg_log_probs = self._generate_rationales(z_probs_) #(batch_size, length)
        
        # masked_input = X_tokens * z.unsqueeze(-1)

        # print("shapes:", masked_input.shape, X_mask.shape)

        predict = self.E_model(X_tokens, attention_mask=z)[0] # the first output are the logits
        # print(predict.shape)
        # print(predict)

        # anti_masked_input = word_embeddings * (1-z).unsqueeze(-1)
        anti_predict = self.E_anti_model(X_tokens, attention_mask=(1-z))[0]

        return predict, anti_predict, cls_predict, z, neg_log_probs

    def get_advantages(self, pred_logits, anti_pred_logits, cls_pred_logits, label, z, neg_log_probs, baseline, mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # supervised loss
        prediction_loss = self.loss_func(cls_pred_logits, label) # (batch_size, )
        sup_loss = torch.mean(prediction_loss)
        
        # total loss of accuracy (not batchwise)
        _, cls_pred = torch.max(cls_pred_logits, dim=1) # (batch_size,)
        _, ver_pred = torch.max(pred_logits, dim=1) # (batch_size,)
        consistency_loss = self.loss_func(pred_logits, cls_pred)
        
        prediction = (ver_pred == label).type(torch.FloatTensor)
        pred_consistency = (ver_pred == cls_pred).type(torch.FloatTensor)
        
        _, anti_pred = torch.max(anti_pred_logits, dim=1)
        prediction_anti = (anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        
        if self.args.cuda:
            prediction = prediction.cuda()  #(batch_size,)
            pred_consistency = pred_consistency.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()

        continuity_loss, sparsity_loss = self.count_regularization_baos_for_both(z, self.count_tokens, self.count_pieces, mask)
        
        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward 
#         rewards = (prediction + pred_consistency) * self.args.lambda_pos_reward - prediction_anti - sparsity_loss - continuity_loss
        if self.game_mode.startswith('3player'):
            rewards = 0.1 * prediction + self.lambda_acc_gap * (prediction - prediction_anti) - sparsity_loss - continuity_loss
        else:
            rewards = prediction - sparsity_loss - continuity_loss
        
        advantages = rewards - baseline # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.args.cuda:
            advantages = advantages.cuda()
        
        return sup_loss, advantages, rewards, pred_consistency, continuity_loss, sparsity_loss
    
    def get_loss(self, pred_logits, anti_pred_logits, cls_pred_logits, label, z, neg_log_probs, baseline, mask):
        reward_tuple = self.get_advantages(pred_logits, anti_pred_logits, cls_pred_logits,
                                           label, z, neg_log_probs, baseline, mask)
        sup_loss, advantages, rewards, consistency_loss, continuity_loss, sparsity_loss = reward_tuple
        
        # (batch_size, q_length)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)
        
        return sup_loss, rl_loss, rewards, consistency_loss, continuity_loss, sparsity_loss


    def train_cls_one_step(self, X_tokens, label, X_mask):
 
        self.opt_G_sup.zero_grad()
        self.generator.classifier.zero_grad()

        # X_tokens, X_mask = self.preprocessor.encode(X_text)
        # word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        # word_embeddings = self.preprocessor.embed(X_tokens, X_mask)
        cls_predict_logits, _, _ = self.generator.classifier(X_tokens, attention_mask=X_mask) # (batch_size, hidden_dim, sequence_length)
        # max_cls_hidden = torch.max(cls_hiddens + (1 - X_mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        # cls_predict = self.generator.Classifier_pred(max_cls_hidden)
        
        sup_loss = torch.mean(self.loss_func(cls_predict_logits, label))
        
        losses = {'g_sup_loss':sup_loss.cpu().data}
        
        sup_loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)

        self.opt_G_sup.step()
        
        return losses, cls_predict_logits

    def train_gen_one_step(self, x, label, mask):
        z_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
        if self.args.cuda:
            z_baseline = z_baseline.cuda()
        
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(x, mask)
        
#         e_loss = torch.mean(self.loss_func(predict, label))
        _, cls_pred = torch.max(cls_predict, dim=1) # (batch_size,)
#         e_loss = torch.mean(self.loss_func(predict, cls_pred)) # e_loss comes from only consistency
        e_loss = (torch.mean(self.loss_func(predict, label)) + torch.mean(self.loss_func(predict, cls_pred))) / 2
        
        # g_sup_loss comes from only cls pred loss
        _, g_rl_loss, z_rewards, consistency_loss, continuity_loss, sparsity_loss = self.get_loss(predict, 
                                                                         anti_predict, 
                                                                         cls_predict, label, z, 
                                                                         neg_log_probs, z_baseline, mask)
        
        losses = {'g_rl_loss':g_rl_loss.cpu().data}

        g_rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
    
        z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
        self.z_history_rewards.append(z_batch_reward)
        
        return losses, predict, anti_predict, cls_predict, z, z_rewards, consistency_loss, continuity_loss, sparsity_loss
    
    def forward_cls(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        # word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        z = torch.ones_like(x).type(torch.cuda.FloatTensor)
        # masked_input = word_embeddings * z.unsqueeze(-1)
        
        predict = self.E_model(X_tokens, attention_mask=z)

        return predict

    #new methods

    def generate_data(self, batch):
        # sort for rnn happiness
        batch.sort_values("counts", inplace=True, ascending=False)
        
        x_mask = np.stack(batch["masks"], axis=0)
        # drop all zero columns
        zero_col_idxs = np.argwhere(np.all(x_mask[...,:] == 0, axis=0))
        x_mask = np.delete(x_mask, zero_col_idxs, axis=1)

        x_mat = np.stack(batch["tokens"], axis=0)
        # drop all zero columns
        x_mat = np.delete(x_mat, zero_col_idxs, axis=1)

        y_vec = np.stack(batch[LABEL_COL], axis=0)
        
        batch_x_ = Variable(torch.from_numpy(x_mat)).to(torch.int64)
        batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
        batch_y_ = Variable(torch.from_numpy(y_vec)).to(torch.int64)

        if args.cuda:
            batch_x_ = batch_x_.cuda()
            batch_m_ = batch_m_.cuda()
            batch_y_ = batch_y_.cuda()

        return batch_x_, batch_m_, batch_y_

    def _get_sparsity(self, z, mask):
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)

        sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
        return sparsity_ratio

    def _get_continuity(self, z, mask):
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)
        
        mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
            
        continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
        
        return continuity_ratio

    def display_example(self, x, m, z):
        seq_len = int(m.sum().item())
        ids = x[:seq_len]
        tokens = self.preprocessor.decode_single(ids)

        final = ""
        for i in range(len(tokens)):
            if z[i]:
                final += "[" + tokens[i] + "]"
            else:
                final += tokens[i]
            final += " "
        print(final)

    def convert_ids_to_tokens_glove(self, ids):
        return [reverse_word_vocab[i.item()] for i in ids]

    def convert_ids_to_tokens_bert(self, ids):
        return [reverse_word_vocab[i.item()] for i in ids]

    def test(self, df_test):
        self.eval()
        
        test_batch_size = 200
        accuracy = 0
        anti_accuracy = 0
        sparsity_total = 0

        for i in range(len(df_test)//test_batch_size):
            test_batch = df_test.iloc[i*test_batch_size:(i+1)*test_batch_size]
            batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)
            predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(batch_x_, batch_m_)
            
            # do a softmax on the predicted class probabilities
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            
            accuracy += (y_pred == batch_y_).sum().item()
            anti_accuracy += (anti_y_pred == batch_y_).sum().item()

            # calculate sparsity
            sparsity_ratios = self._get_sparsity(z, batch_m_)
            sparsity_total += sparsity_ratios.sum().item()

        accuracy = accuracy / len(df_test)
        anti_accuracy = anti_accuracy / len(df_test)
        sparsity = sparsity_total / len(df_test)
        print("Test sparsity: ", sparsity)
        print("Test accuracy: ", accuracy, "% Anti-accuracy: ", anti_accuracy)

        rand_idx = random.randint(0, test_batch_size-1)
        # display an example
        print("Gold Label: ", batch_y_[rand_idx].item(), " Pred label: ", y_pred[rand_idx].item())
        self.display_example(batch_x_[rand_idx], batch_m_[rand_idx], z[rand_idx])

        return accuracy, anti_accuracy, sparsity

    def pretrain_classifier(self, df_train, df_test, batch_size, num_iteration=1000, test_iteration=100):
        train_accs = []
        test_accs = []
        best_train_acc = 0.0
        best_test_acc = 0.0
        self.init_optimizers()
        self.init_rl_optimizers()

        for i in tqdm(range(num_iteration)):
            self.generator.classifier.train() # pytorch fn; sets module to train mode

            # sample a batch of data
            batch = df_train.sample(batch_size, replace=True)
            batch_x_, batch_m_, batch_y_ = self.generate_data(batch)

            losses, predict = self.train_cls_one_step(batch_x_, batch_y_, batch_m_)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)

            acc = np.float((y_pred == batch_y_).sum().cpu().data.item()) / batch_size
            train_accs.append(acc)

            if acc > best_train_acc:
                best_train_acc = acc

            if (i+1) % test_iteration == 0:
                self.eval() # set module to eval mode
                test_correct = 0.0
                test_total = 0.0
                test_count = 0

                for test_iter in range(len(df_test)//batch_size):
                    test_batch = df_test.iloc[test_iter*batch_size:(test_iter+1)*batch_size] # TODO: originally used dev batch here?
                    batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)

                    # predict = self.forward_cls(batch_x_, batch_m_
                    # embeddings = self.preprocessor.embed(batch_x_)
                    self.generator.classifier.eval()
                    predict = self.generator.classifier(batch_x_, batch_m_)[0]

                    _, y_pred = torch.max(predict, dim=1)

                    test_correct += np.float((y_pred == batch_y_).sum().cpu().data.item())
                    test_total += batch_size

                    test_count += batch_size

                    test_accs.append(test_correct / test_total)
                
                if test_correct / test_total > best_test_acc:
                    best_test_acc = test_correct / test_total

                avg_train_accs = sum(train_accs[len(train_accs) - 10:len(train_accs)])/10
                print('train:', avg_train_accs, 'best train acc:', best_train_acc)
                print('test:', test_accs[-1], 'best test:', best_test_acc)
    
    
    def fit(self, df_train, df_test, batch_size, num_iteration=40000, test_iteration=200):
        print('training with game mode:', classification_model.game_mode)
        train_accs = []
        best_test_acc = 0.0
        
        self.init_optimizers()
        self.init_rl_optimizers()
        # old_E_anti_weights = self.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy() #TODO remove? this is never used!

        current_datetime = datetime.now().strftime("%m_%d_%y_%H_%M_%S")

        if self.args.fixed_classifier:
            print("freezing generator classifier.")
            self.freeze_bert_classifier(self.generator.classifier, entire=True)

        if self.args.save_best_model:
            model_folder_path = os.path.join(self.args.save_path, self.args.model_prefix + current_datetime + "training_run")
            os.mkdir(model_folder_path)
            log_filepath = os.path.join(model_folder_path, "training_stats.txt")
            logging.basicConfig(filename=log_filepath, filemode='a', level=logging.INFO)
        for i in tqdm(range(num_iteration)):
            self.train()
        
            # sample a batch of data
            batch = df_train.sample(batch_size, replace=True)
            batch_x_, batch_m_, batch_y_ = self.generate_data(batch)

            z_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
            if self.args.cuda:
                z_baseline = z_baseline.cuda()

            losses, predict, anti_predict, cls_predict, z, z_rewards, consistency_loss, continuity_loss, sparsity_loss = classification_model.train_one_step(\
            batch_x_, batch_y_, z_baseline, batch_m_)

            z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
            self.z_history_rewards.append(z_batch_reward)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            acc = np.float((y_pred == batch_y_).sum().cpu().data.item()) / args.batch_size
            train_accs.append(acc)

            if i % test_iteration == 0:
                # avg_train_acc = sum(train_accs[len(train_accs) - 20: len(train_accs)]) / 20
                print("\Avg train accuracy: ", sum(train_accs[len(train_accs)-10:len(train_accs)])/10)
                test_acc, test_anti_acc, test_sparsity = self.test(df_test)
                print('supervised_loss: %.4f, sparsity_loss: %.4f, continuity_loss: %.4f'%(losses['e_loss'], torch.mean(sparsity_loss).cpu().data, torch.mean(continuity_loss).cpu().data))
                if test_acc > best_test_acc:
                    if self.args.save_best_model:
                        print("saving best model and model stats")
                        current_datetime = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
                        # save model
                        torch.save(self.state_dict(), os.path.join(model_folder_path, self.args.model_prefix + current_datetime + ".pth"))
                        # save stats
                        logging.info('best model at time ' + current_datetime)
                        logging.info('sparsity lambda: %.4f'%(self.args.lambda_sparsity))
                        logging.info('highlight percentage: %.4f'%(self.args.highlight_percentage))
                        logging.info('last train acc: %.4f'%(train_accs[-1]))
                        logging.info('last test acc: %.4f, previous best test acc: %.4f, last anti test acc: %.4f'%(test_acc,  best_test_acc, test_anti_acc))
                        logging.info('last test sparsity: %.4f'%test_sparsity)
                        logging.info('supervised_loss: %.4f, sparsity_loss: %.4f, continuity_loss: %.4f'%(losses['e_loss'], torch.mean(sparsity_loss).cpu().data, torch.mean(continuity_loss).cpu().data))
                    best_test_acc = test_acc


class Argument():
    def __init__(self):
        self.model_type = 'RNN'
        self.cell_type = 'GRU'
        self.hidden_dim = 768 # the hidden dim of the intermediate bert layers
        self.embedding_dim = 768 # the dim of the embedded words
        self.kernel_size = 5
        self.layer_num = 1
        self.fine_tuning = False
        self.z_dim = 2
        self.gumbel_temprature = 0.1
        self.cuda = True
        self.batch_size = 20
        self.mlp_hidden_dim = 50
        self.dropout_rate = 0.4
        self.use_relative_pos = True
        self.max_pos_num = 20
        self.pos_embedding_dim = -1
        self.fixed_classifier = True
        self.fixed_E_anti = True
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 0.0
        self.lambda_anti = 1.0
        self.lambda_pos_reward = 0.1
        self.exploration_rate = 0.05
        self.highlight_percentage = 0.3
        self.highlight_count = 8
        self.count_tokens = 8
        self.count_pieces = 4
        self.lambda_acc_gap = 1.2
        self.label_embedding_dim = 400
        self.game_mode = '3player'
        self.margin = 0.2
        self.lm_setting = 'multiple'
        self.lambda_lm = 1.0
        self.ngram = 4
        self.with_lm = False
        self.batch_size_ngram_eval = 5
        self.lr=0.001
        self.pre_trained_model_prefix = 'pre_trained_cls.model'
        self.save_path = os.path.join("..", "models")
        self.model_prefix = "sst2rnpmodel"
        self.save_best_model = False
        self.num_labels = 2
        self.pre_train_cls = True


if __name__=="__main__":
    # procedure from run_sst2_rationale.py
    # training params
    use_cuda = torch.cuda.is_available()
    load_pre_cls = False
    pre_trained_model_prefix = 'pre_trained_cls.model'
    save_path = os.path.join("..", "models")
    model_prefix = "sst2rnpmodel"
    save_best_model = True
    pre_train_cls = True
    num_labels = 2

    glove_path = os.path.join("..", "datasets", "hiloglove.6B.100d.txt")
    COUNT_THRESH = 1
    #DATA_FOLDER = os.path.join("../../sentiment_dataset/data/")
    DATA_FOLDER = os.path.join("../datasets/sst2/")
    LABEL_COL = "label"
    TEXT_COL = "sentence"
    TOKEN_CUTOFF = 50

    def generate_tokens_glove(word_vocab, text):
        indexed_text = [word_vocab[word] if (counts[word] > COUNT_THRESH) else word_vocab["<UNK>"] for word in text.split()]
        pad_length = TOKEN_CUTOFF - len(indexed_text)
        mask = [1] * len(indexed_text) + [0] * pad_length
        
        indexed_text = indexed_text + [word_vocab["<PAD>"]] * pad_length
        
        return np.array(indexed_text), np.array(mask)

    def get_all_tokens_glove(data):
        l = []
        m = []
        counts = []
        for sentence in data:
            token_list, mask = generate_tokens_glove(word_vocab, sentence)
            l.append(token_list)
            m.append(mask)
            counts.append(np.sum(mask))
        tokens = pd.DataFrame({"tokens": l, "mask": m, "counts": counts})
        return tokens

    def build_vocab(df):
        d = {"<PAD>":0, "<UNK>":1}
        counts = {}
        for i in range(len(df)):
            sentence = df.iloc[i][TEXT_COL]
            for word in sentence.split():
                if word not in d:
                    d[word] = len(d)
                    counts[word] = 1
                else:
                    counts[word] += 1
        reverse_d = {v: k for k, v in d.items()}
        return d, reverse_d, counts

    def initial_embedding(word_vocab, embedding_size, embedding_path=None): 
        vocab_size = len(word_vocab)
        # initialize a numpy embedding matrix 
        
        embeddings = 0.1*np.random.randn(vocab_size, embedding_size).astype(np.float32)
        
        # replace the <PAD> embedding by all zero
        embeddings[0, :] = np.zeros(embedding_size, dtype=np.float32)

        if embedding_path and os.path.isfile(embedding_path):
            f = open(embedding_path, "r", encoding="utf8")
            counter = 0
            for line in f:
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = data[1::]
                embedding = list(map(np.float32, embedding))
                if word in word_vocab:
                    embeddings[word_vocab[word], :] = embedding
                    counter += 1
            f.close()
            print("%d words has been switched."%counter)
        else:
            print("embedding is initialized fully randomly.")

        return embeddings

    def load_data(fpath):
        df_dict = {LABEL_COL: [], TEXT_COL: []}
        with open(fpath, 'r') as f:
            label_start = 0
            sentence_start = 2
            for line in f:
                label = int(line[label_start])
                sentence = line[sentence_start:]
                df_dict[LABEL_COL].append(label)
                df_dict[TEXT_COL].append(sentence)
        return pd.DataFrame.from_dict(df_dict)


    # df_train = load_data(os.path.join(DATA_FOLDER, 'stsa.binary.train'))
    # df_test = load_data(os.path.join(DATA_FOLDER, 'stsa.binary.test'))
    # # TODO combine train and test dataset into df_all
    # df_all = pd.concat([df_train, df_test])

    # word_vocab, reverse_word_vocab, counts = build_vocab(df_all)
    # embeddings = initial_embedding(word_vocab, 100, glove_path)

    # # create training and testing labels
    # y_train = df_train[LABEL_COL]
    # y_test = df_test[LABEL_COL]

    # # create training and testing inputs
    # X_train = df_train[TEXT_COL]
    # X_test = df_test[TEXT_COL]

    # df_train = pd.concat([df_train, get_all_tokens_glove(X_train)], axis=1)
    # df_test = pd.concat([df_test, get_all_tokens_glove(X_test)], axis=1)

    print("loading data...")
    df_train = load_data(os.path.join(DATA_FOLDER, 'stsa.binary.train')) #TODO back to train after bug fixes
    df_test = load_data(os.path.join(DATA_FOLDER, 'stsa.binary.test'))

    # preprocess to generate tokens, masks, and token counts per training entry
    print("tokenizing and preprocessing training data...")
    bp = BertPreprocessor(max_length=TOKEN_CUTOFF)
    input_ids, masks, counts = bp.encode(df_train[TEXT_COL])
    df_train_data = pd.DataFrame({"tokens": input_ids, "masks": masks, "counts": counts})
    print(len(df_train_data), " training samples.")
    print("tokenizing and preprocessing testing data...")
    input_ids, masks, counts = bp.encode(df_test[TEXT_COL])
    df_test_data = pd.DataFrame({"tokens": input_ids, "masks": masks, "counts": counts})
    print(len(df_test_data), " testing samples.")

    df_train = pd.concat([df_train, df_train_data], axis=1)
    df_test = pd.concat([df_test, df_test_data], axis=1)

    # TODO: phase out use of args
    args = Argument()
    print(vars(args))
    # get embeddings, number of labels

    classification_model = ThreePlayerModel(args, bp, num_labels)
    if args.cuda:
        classification_model.cuda()
    print(classification_model)

    # optionally pre train classifier
    if args.pre_train_cls:
        print('pre-training the classifier')
        classification_model.pretrain_classifier(df_train, df_test, args.batch_size)

    # train the model
    classification_model.fit(df_train, df_test, args.batch_size)
