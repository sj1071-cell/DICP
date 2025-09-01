import torch
import torch.nn as nn


class NodeModel(nn.Module):

    def __init__(self, args):
        super(NodeModel, self).__init__()
        self.args = args

    def forward(self, graph_embed, graph_event1, graph_event1_mask, graph_event2, graph_event2_mask):
        # model forward propagation
        graph_event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(graph_embed, graph_event1)],
                                 dim=0)
        graph_event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(graph_embed, graph_event2)],
                                 dim=0)

        graph_m1 = graph_event1_mask.unsqueeze(-1).expand_as(graph_event1).float()
        graph_m2 = graph_event2_mask.unsqueeze(-1).expand_as(graph_event2).float()

        graph_event1 = graph_event1 * graph_m1
        graph_event2 = graph_event2 * graph_m2

        # mean
        opt1 = torch.sum(graph_event1, dim=1)
        opt2 = torch.sum(graph_event2, dim=1)

        opt1_len = torch.count_nonzero(graph_event1_mask, dim=1).view(-1, 1)
        opt2_len = torch.count_nonzero(graph_event2_mask, dim=1).view(-1, 1)
        opt1_len[opt1_len < 1] = 1
        opt2_len[opt2_len < 1] = 1
        opt1 = (1.0 / opt1_len) * opt1
        opt2 = (1.0 / opt2_len) * opt2

        node_opt = torch.add(opt1, opt2)

        return node_opt
