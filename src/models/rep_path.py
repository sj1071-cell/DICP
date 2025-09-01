import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class PathModel(nn.Module):

    def __init__(self, args):
        super(PathModel, self).__init__()
        self.args = args

        self.rel_dim = args.rgcn_hidden_size
        self.rel_embeds = nn.Parameter(torch.Tensor(args.num_rels * 2, self.rel_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

        self.rnn_layers = args.rnn_layers
        self.rnn_hidden_size = args.rnn_hidden_size
        self.bidirectional = args.rnn_bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn_opt_dim = self.rnn_hidden_size * self.num_directions

        self.rnn = nn.LSTM(self.rel_dim * 2, self.rnn_hidden_size, self.rnn_layers,
                           batch_first=True, bidirectional=self.bidirectional)
        self.num_heads = 1
        self.multihead_attn = nn.MultiheadAttention(self.rnn_opt_dim, self.num_heads)

        self.pad_path_embed = nn.Parameter(torch.Tensor(1, self.rnn_opt_dim))
        nn.init.xavier_uniform_(self.pad_path_embed, gain=nn.init.calculate_gain('relu'))

    def forward(self, graph_embed, paths_list, paths_len_list, node_opt):
        # get path vector
        assert len(graph_embed) == len(paths_len_list)
        batch_sample_path = []
        start_idx = 0
        for paths_num, graph_embed_sample, node_opt_sample in zip(paths_len_list, graph_embed, node_opt):
            end_idx = start_idx + paths_num
            sample_paths = paths_list[start_idx:end_idx]
            start_idx = end_idx

            sample_paths_tensor_list = []
            sample_paths_len = []
            for path in sample_paths:
                path_tensor_list = []
                for item in path:
                    rel = item[1]
                    node = torch.LongTensor([item[0]]).to(self.args.device)
                    if rel == 'Padr':
                        node_tensor = torch.index_select(graph_embed_sample, 0, node)
                        node_tensor = F.pad(node_tensor, (0, self.rel_dim))
                        path_tensor_list.append(node_tensor)
                    else:
                        node_tensor = torch.index_select(graph_embed_sample, 0, node)
                        rel = torch.LongTensor([rel]).to(self.args.device)
                        rel_tensor = torch.index_select(self.rel_embeds, 0, rel)
                        path_tensor_list.append(node_tensor)
                        path_tensor_list.append(rel_tensor)

                path_tensor = torch.cat(path_tensor_list, dim=1)
                path_tensor = path_tensor.reshape(-1, self.rel_dim * 2)
                sample_paths_tensor_list.append(path_tensor)
                sample_paths_len.append(path_tensor.shape[0])

            if len(sample_paths) > 0:
                sample_paths_tensor = rnn_utils.pad_sequence(sample_paths_tensor_list, batch_first=True)
                sample_paths_pack = rnn_utils.pack_padded_sequence(sample_paths_tensor, sample_paths_len,
                                                                   batch_first=True, enforce_sorted=False)
                rnn_opt, (h_n, c_n) = self.rnn(sample_paths_pack)
                out_pad, out_len = rnn_utils.pad_packed_sequence(rnn_opt, batch_first=True)

                out_list = []
                for out_tensor, index in zip(out_pad, out_len):
                    out_list.append(out_tensor[index - 1].reshape(1, -1))
                rnn_tensor = torch.cat(out_list, dim=0).unsqueeze(1)

                # attention
                query = node_opt_sample.reshape(1, 1, -1)
                key = rnn_tensor
                value = rnn_tensor
                attn_output, attn_output_weights = self.multihead_attn(query, key, value)
                sample_opt = attn_output.squeeze(1)
            else:
                sample_opt = self.pad_path_embed
            batch_sample_path.append(sample_opt)
        path_opt = torch.cat(batch_sample_path, dim=0)

        return path_opt
