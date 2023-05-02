import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
from data_utils_bert import Tokenizer4Bert, pad_and_truncate
from models import BERTGCN
from graph import load_sentic_word, sentic_dependency_adj_matrix, dependency_adj_matrix


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        fname = {
            'headlines': {
                'train': './datasets/headlines/test.raw',
                'test': './datasets/headlines/train.raw'
            },
            'riloff': {
                'train': './datasets/riloff/test.raw',
                'test': './datasets/riloff/train.raw'
            },
        }

        self.tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = self.opt.model_class(bert, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        senticNet = load_sentic_word()

        text_indices = self.tokenizer.text_to_sequence(raw_text.lower())
        text_len = np.sum(text_indices != 0)

        concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + raw_text.lower() + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        dependency_graph = dependency_adj_matrix(raw_text.lower(), senticNet)
        dependency_graph = np.pad(dependency_graph, \
                            ((0, self.tokenizer.max_seq_len-dependency_graph.shape[0]), (0, self.tokenizer.max_seq_len - dependency_graph.shape[0])), 'constant')
        dependency_graph = torch.tensor([dependency_graph])
        
        sdat_graph = sentic_dependency_adj_matrix(raw_text.lower(), senticNet)
        sdat_graph = np.pad(sdat_graph, \
                    ((0,self.tokenizer.max_seq_len-sdat_graph.shape[0]),(0,self.tokenizer.max_seq_len-sdat_graph.shape[0])), 'constant')
        sdat_graph = torch.tensor([sdat_graph])

        data = {
            'text_bert_indices': concat_bert_indices,
            'bert_segments_indices': concat_segments_indices,
            'text_indices': text_indices,
            'dependency_grap': dependency_graph,
            'affective_graph': sdat_graph,
        }

        t_inputs = [data[col].to(opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs



if __name__ == '__main__':
    dataset = 'headlines'
    # set your trained models here
    model_state_dict_paths = {
        'bertgcn': './saved_models/bertgcn_'+dataset+'.pkl',
    }
    model_classes = {
        'bertgcn': BERTGCN,
    }
    input_colses = {
        'bertgcn': ['text_bert_indices', 'text_indices', 'aspect_indices', 'bert_segments_indices', 'left_indices', 'sdat_graph'],
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = 'bertgcn'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.bert_dim = 768
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.max_seq_len = 85
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_text = 'Food is always fresh and hot - ready to eat !'
    aspect = 'food'

    print('The input are as follows:')
    print('Sentence:', raw_text)
    print('Aspect:', aspect)

    inf = Inferer(opt)

    print('='*10, 'Inferring ......')

    t_probs = inf.evaluate(raw_text)
    infer_label = t_probs.argmax(axis=-1)[0] - 1
    label_dict = {-1: 'Non-Sarcastic', 1: 'Sarcastic'}

    print('The test results is:', infer_label, label_dict[infer_label])
