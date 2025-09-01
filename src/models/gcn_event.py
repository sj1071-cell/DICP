import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_model import RGCN
from .rep_node import NodeModel
from .rep_path import PathModel


class GCNEvent(nn.Module):

    def __init__(self, args):
        super(GCNEvent, self).__init__()
        self.model_gcn = RGCN(args, args.rgcn_hidden_size, args.rgcn_dropout,
                              args.num_ents, args.num_rels, args.rgcn_bases, args.rgcn_layers,
                              args.rgcn_name, self_loop=True, skip_connect=True)
        self.model_node = NodeModel(args)
        self.model_path = PathModel(args)

        self.num_directions = 2 if args.rnn_bidirectional else 1
        self.dropout = nn.Dropout(args.h_dropout)
        # self.classifier = nn.Linear(768 + args.rgcn_hidden_size * 1 + args.rnn_hidden_size * self.num_directions,
        #                             args.num_labels)

    def forward(self, graphs, graphs_len, graph_event1, graph_event1_mask, graph_event2, graph_event2_mask,
                paths_list, paths_len_list, bert_init_list, random_init_list, sent_h):
        # get node opt
        graph_embed, _ = self.model_gcn.forward(graphs, graphs_len, bert_init_list, random_init_list, sent_h)
        node_opt = self.model_node.forward(graph_embed, graph_event1, graph_event1_mask,
                                           graph_event2, graph_event2_mask)
        # get path opt
        path_opt = self.model_path.forward(graph_embed, paths_list, paths_len_list,
                                           node_opt)

        gcn_opt = torch.cat([node_opt, path_opt], dim=1)

        return gcn_opt