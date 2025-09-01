import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None, skip_connect=False,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect

        if self.skip_connect:
            # Not the same as a self-loop, it's a cross-layer computation
            self.skip_connect_weight = nn.Parameter(torch.DoubleTensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.DoubleTensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # Initialization set to 0

        if self.bias:
            self.bias = nn.Parameter(torch.DoubleTensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.DoubleTensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h):
        # The function has inconsistent results in each experiment, it may be caused by the torch.mm function.
        # So first convert the precision to double, then convert back to float.
        # Similarly, weight should also be double when initialized.
        g.ndata['h'] = g.ndata['h'].double()
        g.ndata['norm'] = g.ndata['norm'].double()

        if len(prev_h) != 0 and self.skip_connect:
            prev_h = prev_h.double()
            skip_weight = torch.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
            # use sigmoid, let value between 0 and 1

        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias

        if len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)

        node_repr = node_repr.half()
        node_repr = node_repr.float()
        g.ndata['h'] = node_repr
        g.ndata['norm'] = g.ndata['norm'].float()

        return node_repr


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, dropout=0.0, self_loop=False, skip_connect=False, rel_emb=None):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, skip_connect, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.DoubleTensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

