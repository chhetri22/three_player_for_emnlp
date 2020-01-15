import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque


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
                                    num_layers=args.layer_num, bidirectional=True)
        elif args.cell_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=input_dim, 
                                     hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True)
    
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
    
class Generator(nn.Module):
    
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
        super(Generator, self).__init__()
        
        self.args = args
        self.z_dim = args.z_dim
        
        if args.model_type == "CNN":
            self.generator_model = CnnModel(args)
        elif args.model_type == "RNN":
            self.generator_model = RnnModel(args, input_dim)
        self.output_layer = nn.Linear(args.hidden_dim, self.z_dim)
        
    def forward(self, x, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        
        #(batch_size, sequence_length, hidden_dim)
        hiddens = self.generator_model(x, mask).transpose(1, 2).contiguous() 
        scores = self.output_layer(hiddens) # (batch_size, sequence_length, 2)
        return scores


class ThreePlayerModel(nn.Module):
    """flattening the HardRationale3PlayerClassificationModelForEmnlp -> HardRationale3PlayerClassificationModel -> 
       Rationale3PlayerClassificationModel dependency structure from original paper code"""

    def __init__(self, args, embeddings, num_labels, explainer=ClassifierModule, anti_explainer=ClassifierModule, generator=Generator, classifier=ClassifierModule):
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
        self.E_model = explainer(args)
        self.E_anti_model = anti_explainer(args)
        self.C_model = classifier(args)
        self.generator = generator(args, self.input_dim)
                    
        # Independent inputs: embeddings, num_labels
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed_layer.weight.data = torch.from_numpy(embeddings)
        self.embed_layer.weight.requires_grad = self.args.fine_tuning
        self.num_labels = num_labels 
  
        # no internal code dependencies
        self.NEG_INF = -1.0e6 # TODO: move out and set as constant?
        self.loss_func = nn.CrossEntropyLoss(reduce=False)


    # methods from Hardrationale3PlayerClassificationModel
    def init_optimizers(self): # not sure if this can be merged with initializer
        self.opt_E = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_model.parameters()), lr=self.args.lr)
        self.opt_E_anti = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_anti_model.parameters()), lr=self.args.lr)
    
    def init_rl_optimizers(self):
        self.opt_G_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.args.lr * 0.1)
    
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

    def train_one_step(self, x, label, baseline, mask):
        predict, anti_predict, z, neg_log_probs = self.forward(x, mask)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, z, 
                                                                     neg_log_probs, baseline, 
                                                                     mask, label)
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        return losses, predict, anti_predict, z, rewards, continuity_loss, sparsity_loss
        
    def forward(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)

        z_scores_ = self.generator(word_embeddings, mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - mask.unsqueeze(-1)) * z_probs_)
        
        z, neg_log_probs = self._generate_rationales(z_probs_)
        
        predict = self.E_model(word_embeddings, z, mask)
        
        anti_predict = self.E_anti_model(word_embeddings, 1 - z, mask)

        return predict, anti_predict, z, neg_log_probs

    def get_advantages(self, predict, anti_predict, label, z, neg_log_probs, baseline, mask, x=None):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # total loss of accuracy (not batchwise)
        _, y_pred = torch.max(predict, dim=1)
        prediction = (y_pred == label).type(torch.FloatTensor)
        _, y_anti_pred = torch.max(anti_predict, dim=1)
        prediction_anti = (y_anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        if self.args.cuda:
            prediction = prediction.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
        continuity_loss, sparsity_loss = self.count_regularization_baos_for_both(z, self.args.count_tokens, self.args.count_pieces, mask)
        
        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward
        if self.game_mode.startswith('3player'):
            # rewards = prediction - prediction_anti - sparsity_loss - continuity_loss
            rewards = 0.1 * prediction + self.lambda_acc_gap * (prediction - prediction_anti) - sparsity_loss - continuity_loss
        else:
            rewards = prediction - sparsity_loss - continuity_loss
        
        advantages = rewards - baseline # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.args.cuda:
            advantages = advantages.cuda()
        
        if x is None:
            return advantages, rewards, continuity_loss, sparsity_loss

    def get_loss(self, predict, anti_predict, z, neg_log_probs, baseline, mask, label, x=None):
        reward_tuple = self.get_advantages(predict, anti_predict, label, z, neg_log_probs, baseline, mask, x)
        
        if x is None:
            advantages, rewards, continuity_loss, sparsity_loss = reward_tuple
        else:
            advantages, rewards, continuity_loss, sparsity_loss, lm_prob = reward_tuple
        
        # (batch_size, q_length)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)
        
        if x is None:
            return rl_loss, rewards, continuity_loss, sparsity_loss
        else:
            return rl_loss, rewards, continuity_loss, sparsity_loss, lm_prob

    def train_cls_one_step(self, x, label, mask):
        predict = self.forward_cls(x, mask)
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        losses = {'e_loss':e_loss.cpu().data}
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        return losses, predict
    
    def forward_cls(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        z = torch.ones_like(x).type(torch.cuda.FloatTensor)
        
        predict = self.E_model(word_embeddings, z, mask)

        return predict

    #new methods

    def generate_data(self, batch):
        # sort for rnn happiness
        batch.sort_values("counts", inplace=True, ascending=False)
        
        x_mask = np.stack(batch["mask"], axis=0)
        # drop all zero columns
        zero_col_idxs = np.argwhere(np.all(x_mask[...,:] == 0, axis=0))
        x_mask = np.delete(x_mask, zero_col_idxs, axis=1)

        x_mat = np.stack(batch["tokens"], axis=0)
        # drop all zero columns
        x_mat = np.delete(x_mat, zero_col_idxs, axis=1)

        y_vec = np.stack(batch["label"], axis=0)
        
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
        tokens = self.convert_ids_to_tokens_glove(ids)

        final = ""
        for i in range(len(m)):
            if z[i]:
                final += "[" + tokens[i] + "]"
            else:
                final += tokens[i]
            final += " "
        print(final)

    def convert_ids_to_tokens_glove(self, ids):
        return [reverse_word_vocab[i.item()] for i in ids]

    def test(self):
        self.eval()
        
        test_size = 200
        
        test_batch = df_test.sample(test_size)
        batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)
        predict, anti_predict, z, neg_log_probs = self.forward(batch_x_, batch_m_)
        
        # do a softmax on the predicted class probabilities
        _, y_pred = torch.max(predict, dim=1)
        _, anti_y_pred = torch.max(anti_predict, dim=1)
        
        # calculate sparsity
        sparsity_ratios = self._get_sparsity(z, batch_m_)
        print("Test sparsity: ", sparsity_ratios.sum().item() / len(sparsity_ratios))
        
        accuracy = (y_pred == batch_y_).sum().item() / test_size
        anti_accuracy = (anti_y_pred == batch_y_).sum().item() / test_size
        print("Test accuracy: ", accuracy, "% Anti-accuracy: ", anti_accuracy)

        # display an example
        print("Gold Label: ", batch_y_[0].item(), " Pred label: ", y_pred[0].item())
        self.display_example(batch_x_[0], batch_m_[0], z[0])

    def pretrain_classifier(self, df_train, batch_size, num_iteration=2000, display_iteration=10, test_iteration=10):
        train_accs = []
        # best_dev_acc = 0.0

        self.init_optimizers() # ?? do we have to do this here?

        for i in tqdm(range(num_iteration)):
            self.train() # pytorch fn; sets module to train mode

            # sample a batch of data
            batch = df_train.sample(batch_size, replace=True)
            batch_x_, batch_m_, batch_y_ = self.generate_data(batch)

            losses, predict = self.train_cls_one_step(batch_x_, batch_y_, batch_m_)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)

            acc = np.float((y_pred == batch_y_).sum().cpu().data[0]) / batch_size
            train_accs.append(acc)

            if (i+1) % test_iteration == 0:
                self.eval() # set module to eval mode

                test_batch = df_test.sample(batch_size) # TODO: originally used dev batch here?
                batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)

                predict = self.forward_cls(batch_x_, batch_m_)


                eval_sets = ['dev']
                for set_name in (eval_sets):
                    dev_correct = 0.0
                    dev_anti_correct = 0.0
                    dev_cls_correct = 0.0
                    dev_total = 0.0
                    sparsity_total = 0.0
                    dev_count = 0

                    num_dev_instance = df_train.data_sets[set_name].size()

                    for start in range(num_dev_instance // args.batch_size):
                        dev_batch = df_dev.sample(batch_size)
                        batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)

                        predict = self.forward_cls(batch_x_, batch_m_)
                        # calculate classification accuarcy
                        _, y_pred = torch.max(predict, dim=1)

                        dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
                        dev_total += batch_size

                        dev_count += batch_size

                    if set_name == 'dev':
                        dev_accs.append(dev_correct / dev_total)
                        dev_anti_accs.append(dev_anti_correct / dev_total)
                        dev_cls_accs.append(dev_cls_correct / dev_total)
                        if dev_correct / dev_total > best_dev_acc:
                            best_dev_acc = dev_correct / dev_total

                    else:
                        test_accs.append(dev_correct / dev_total)

                print('train:', train_accs[-1])
                print('dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count)
    
    
    def fit(self, df_train, batch_size, num_iteration=80000, display_iteration=10, test_iteration=10):
        print('training with game mode:', classification_model.game_mode)
        train_accs = []

        num_iteration = 20000
        display_iteration = 1
        test_iteration = 50
        
        self.init_optimizers()
        self.init_rl_optimizers()
        old_E_anti_weights = self.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()

        for i in tqdm(range(num_iteration)):
            self.train()
        
            # sample a batch of data
            batch = df_train.sample(batch_size, replace=True)
            batch_x_, batch_m_, batch_y_ = self.generate_data(batch)

            z_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
            if self.args.cuda:
                z_baseline = z_baseline.cuda()

            losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = classification_model.train_one_step(\
            batch_x_, batch_y_, z_baseline, batch_m_)

            z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
            self.z_history_rewards.append(z_batch_reward)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)

            acc = np.float((y_pred == batch_y_).sum().cpu().data.item()) / args.batch_size
            train_accs.append(acc)

            if i % test_iteration == 0:
                self.test()


class Argument():
    def __init__(self):
        self.model_type = 'RNN'
        self.cell_type = 'GRU'
        self.hidden_dim = 400
        self.embedding_dim = 100
        self.kernel_size = 5
        self.layer_num = 1
        self.fine_tuning = False
        self.z_dim = 2
        self.gumbel_temprature = 0.1
        self.cuda = True
        self.batch_size = 40
        self.mlp_hidden_dim = 50
        self.dropout_rate = 0.4
        self.use_relative_pos = True
        self.max_pos_num = 20
        self.pos_embedding_dim = -1
        self.fixed_classifier = True
        self.fixed_E_anti = True
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 1.0
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
        self.save_best_model = True
        self.num_labels = 2


if __name__=="__main__":
    # procedure from run_sst2_rationale.py
    # training params
    use_cuda = torch.cuda.is_available()
    batch_size = 40
    load_pre_cls = False
    pre_trained_model_prefix = 'pre_trained_cls.model'
    save_path = os.path.join("..", "models")
    model_prefix = "sst2rnpmodel"
    save_best_model = True
    pre_train_cls = False
    num_labels = 2

    glove_path = os.path.join("..", "datasets", "glove.6B.100d.txt")
    COUNT_THRESH = 3
    DATA_FOLDER = os.path.join("../../data/sst2/")
    LABEL_COL = "label"
    TEXT_COL = "sentence"
    TOKEN_CUTOFF = 70

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


    df_train = load_data(os.path.join(DATA_FOLDER, 'stsa.binary.train'))
    df_test = load_data(os.path.join(DATA_FOLDER, 'stsa.binary.test'))
    # TODO combine train and test dataset into df_all
    df_all = pd.concat([df_train, df_test])

    word_vocab, reverse_word_vocab, counts = build_vocab(df_all)
    embeddings = initial_embedding(word_vocab, 100, glove_path)

    # create training and testing labels
    y_train = df_train[LABEL_COL]
    y_test = df_test[LABEL_COL]

    # create training and testing inputs
    X_train = df_train[TEXT_COL]
    X_test = df_test[TEXT_COL]

    df_train = pd.concat([df_train, get_all_tokens_glove(X_train)], axis=1)
    df_test = pd.concat([df_test, get_all_tokens_glove(X_test)], axis=1)

    # TODO: phase out use of args
    args = Argument()
    print(vars(args))
    # get embeddings, number of labels
    classification_model = ThreePlayerModel(args, embeddings, num_labels)
    if use_cuda:
        classification_model.cuda() 
    print(classification_model)

    # optionally pre train classifier
    if pre_train_cls:
        print('pre-training the classifier')
        classification_model.pretrain_classifier(data, batch_size)

    # train the model
    classification_model.fit(df_train, batch_size)