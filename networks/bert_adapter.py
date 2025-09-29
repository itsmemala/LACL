#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F
sys.path.append("./networks/base/")
from .my_transformers import MyBertModel

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter = True
        self.bert = MyBertModel.from_pretrained(args.bert_model,config=config,args=args)

        #BERT fixed all ===========
        for param in self.bert.parameters():
            # param.requires_grad = True
            param.requires_grad = False

        #But adapter is open

        #Only adapters are trainable

        if args.apply_bert_output and args.apply_bert_attention_output:
            adaters = \
                [self.bert.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_bert_output:
            adaters = \
                [self.bert.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif self.apply_bert_attention_output:
            adaters = \
                [self.bert.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)]


        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True
                # param.requires_grad = False

        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        if 'dil' in args.scenario:
            # self.last=torch.nn.Linear(args.bert_hidden_size,args.nclasses)
            self.last=torch.nn.Linear(args.bert_hidden_size,self.taskcla[0][1]) # Infer from the first task
            print('\nDIL with '+str(self.taskcla[0][1])+' classes\n')
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(args.bert_hidden_size,n))
        elif 'cil' in args.scenario:
            self.last=torch.nn.Linear(args.bert_hidden_size,sum([n for t,n in self.taskcla]))
        
        self.classifier = self.classifier_cil if 'cil' in args.scenario else self.classifier_all


        print('BERT ADAPTER')

        return

    def forward(self,input_ids, segment_ids, input_mask, fa_method=None, tid=None):
        # output_dict = {}


        # sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        # pooled_output = self.dropout(pooled_output)
        # #shared head

        # if 'dil' in self.args.scenario:
            # y = self.last(pooled_output)
        # elif 'til' in self.args.scenario:
            # y=[]
            # for t,i in self.taskcla:
                # y.append(self.last[t](pooled_output))
        # elif 'cil' in self.args.scenario:
            # y = self.last(pooled_output)

        # output_dict['y'] = y
        # output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)
        
        # if fa_method=='ig':
            # # print(input_ids.shape)
            # # print(pooled_output.shape)
            # # print(output_dict['y'][tid].shape)
            # return output_dict['y'][tid]
        
        features = self.features(input_ids, segment_ids, input_mask)
        output_dict = self.classifier(features,fa_method,tid)

        return output_dict
    
    def features(self,input_ids, segment_ids, input_mask):
        
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        pooled_output = self.dropout(pooled_output)

        return pooled_output
    
    def classifier_all(self,pooled_output, fa_method=None, tid=None):
        output_dict = {}


        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](pooled_output))
        elif 'cil' in self.args.scenario:
            y = self.last(pooled_output)

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)
        
        if fa_method=='ig':
            # print(input_ids.shape)
            # print(pooled_output.shape)
            # print(output_dict['y'][tid].shape)
            return output_dict['y'][tid]

        return output_dict
    
    def classifier_cil(self,pooled_output, fa_method=None, tid=None):
        output_dict = {}


        y = self.last(pooled_output)

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)
        
        # if fa_method=='ig':
            # # print(input_ids.shape)
            # # print(pooled_output.shape)
            # # print(output_dict['y'][tid].shape)
            # return output_dict['y'][tid]

        return output_dict
    