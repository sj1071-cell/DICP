import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_base import BaseRGCN
from .rgcn_layer import RGCNBlockLayer


class RGCNCell(BaseRGCN):
    def __init__(self, init_dict, num_nodes, h_dim, out_dim, num_rels, num_bases, num_layers, dropout, self_loop,
                 skip_connect, model, rel_embeds):
        super(RGCNCell, self).__init__(num_nodes, h_dim, out_dim, num_rels, num_bases, num_layers, dropout, self_loop,
                 skip_connect, model, rel_embeds)
        self.init_dict = init_dict

    def build_hidden_layer(self, idx):
        act = F.relu if idx != self.num_hidden_layers - 1 else None
        if idx:
            self.num_basis = 0
        # print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "rgcn":
            return RGCNBlockLayer(self.h_dim, self.h_dim, 2 * self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc,
                                  rel_emb=self.rel_emb)

    def forward(self, g, init_rel_emb):
        prev_h = []
        r_weight = None
        if self.skip_connect:
            for i, layer in enumerate(self.layers):
                prev_h = layer(g, prev_h)
        else:
            for layer in self.layers:
                r_weight = layer(g, [])
        return g.ndata.pop('h'), r_weight


class RGCN(nn.Module):
    def __init__(self, args, h_dim, dropout, num_nodes, num_rels, num_bases, num_layers, model,
                 self_loop, skip_connect):
        super(RGCN, self).__init__()
        self.args = args
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.model = model
        self.dropout = nn.Dropout(dropout)

        self.init_dict = self.get_init_dict(args)

        self.node_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))
        nn.init.xavier_uniform_(self.node_embeds, gain=nn.init.calculate_gain('relu'))

        if self.model == "rgcn" or self.model == "gcn":
            self.rel_embeds = None
        else:
            self.rel_embeds = nn.Parameter(torch.Tensor(num_rels * 2, h_dim))
            nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

        self.rgcn = RGCNCell(self.init_dict,
                             self.num_nodes,
                             h_dim,
                             h_dim,
                             num_rels,
                             num_bases,
                             self.num_layers,
                             dropout,
                             self.self_loop,
                             self.skip_connect,
                             self.model,
                             self.rel_embeds)

    def get_init_dict(self, args):
        file_path = os.path.join(args.project_path, args.vertex_dict)

        map_dict = {}
        node_dict = {}
        with open(file_path, 'r') as f:
            for index, line in enumerate(f.readlines()):
                line = line.rstrip('\n')
                if len(line.split('__')) > 1:
                    map_index = node_dict[line.split('__')[0]]
                    node_dict[line] = map_index
                    map_dict[index] = map_index
                else:
                    node_dict[line] = index
                    map_dict[index] = index

        return map_dict

    def init_embed(self, graphs, bert_init_list, random_init_list, sent_h):
        init_graphs = []
        for g_index, (g, sent_h_item) in enumerate(zip(graphs, sent_h)):
            init_embed = []
            node_id = g.ndata['id'].squeeze()
            for id in range(len(node_id)):
                if id in bert_init_list[g_index].keys():
                    bert_index = bert_init_list[g_index][id]
                    embed_len = len(bert_index)
                    bert_index = torch.LongTensor(bert_index).to(self.args.device)
                    embed = torch.cat([sent_h_item[i, :].unsqueeze(0) for i in bert_index], dim=0)
                    embed = torch.sum(embed, dim=0, keepdim=True) / embed_len
                elif id in random_init_list[g_index].keys():
                    random_index = random_init_list[g_index][id]
                    # init same node with same rep, map with dict
                    node_id = self.init_dict[random_index]
                    embed = self.node_embeds[node_id].unsqueeze(0)
                init_embed.append(embed)

            init_g = g.to(self.args.device)
            init_embed = torch.cat(init_embed, dim=0)
            init_g.ndata['h'] = init_embed
            init_graphs.append(init_g)

        return init_graphs

    def forward(self, graphs, graphs_length, bert_init_list, random_init_list, sent_h):
        init_graphs = self.init_embed(graphs, bert_init_list, random_init_list, sent_h)
        batched_graphs = dgl.batch(init_graphs)
        embeds, r_weight = self.rgcn.forward(batched_graphs, self.rel_embeds)

        embeds = torch.split(embeds, graphs_length)

        return embeds, r_weight