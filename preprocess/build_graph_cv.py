import os
import pickle
import argparse
import torch
import dgl
import numpy as np
import networkx as nx


def build_sub_graph(num_nodes, num_rels, triples):
    """
    build subgraph as dgl object
    :param num_nodes: the number of node in the graph
    :param num_rels: the number of relations in the graph
    :param triples: the triples in the graph
    :return:
    """

    def comp_deg_norm(g):
        # indegrees normaliztion, if indegree is 0, set indegree is 1
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))  # add reverse, the edge is bidirectional
    rel = np.concatenate((rel, rel + num_rels))

    # edges is indices to reconstruct array
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))

    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    g.add_edges(src, dst)  # node id is converted to index, index -> index is edges
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    g.ids = {}
    for idx, idd in enumerate(uniq_v):
        g.ids[idd] = idx
    # if use_cuda:
    #     g.to(gpu)
    return g


def load_amr_graphs(ent2id, rel2id, sample_triple):
    num_ents = len(ent2id.keys())
    num_rels = len(rel2id.keys())

    triples = []
    for line in sample_triple.split('\n'):
        try:
            h, r, t = eval(line.strip())
            triples.append((h, r, t))
        except:
            print('Read triple data error!')
            print(line)
    graph = build_sub_graph(num_ents, num_rels, np.array(triples))

    return graph


def load_dict(file_path):
    with open(file_path, 'r') as f:
        dict_ = {line.rstrip('\n'): idx for idx, line in enumerate(f.readlines())}

    return dict_


def build_path_graph(rel2id, sample_triple):
    num_rels = len(rel2id.keys())

    g = nx.MultiDiGraph()
    for line in sample_triple.split('\n'):
        try:
            node, relation, next_node = eval(line.strip())
            g.add_edge(node, next_node, key=relation)
            g.add_edge(next_node, node, key=relation + num_rels)
        except:
            print('Read triple data error!')
            print(line)

    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Causal-TimeBank")
    args = parser.parse_args()

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data', args.dataset_name)

    dir_list = ['amr_sample/amr_graph/10fold', 'amr_sample/amr_path/10fold', 'amr_sample/amr_align/10fold']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    text_dict = {}
    with open(os.path.join(data_path, 'tokenized_data/sentences.txt'), 'r') as f:
        res = f.read()
        res = res.rstrip('\n')
        text_list = res.split('\n')
    for idx, text in enumerate(text_list):
        text_dict[text] = idx

    ent2id = load_dict(os.path.join(data_path, 'amr_sample/amr_dict/vertex_dict.txt'))
    rel2id = load_dict(os.path.join(data_path, 'amr_sample/amr_dict/edge_dict.txt'))

    with open(os.path.join(data_path, 'amr_sample/amr_triple/amr_triple.txt'), 'r') as f:
        res = f.read().rstrip('\n\n\n')
        amr_triple = res.split('\n\n\n')
    with open(os.path.join(data_path, 'amr_sample/amr_align/align_data.pkl'), 'rb') as f:
        align_data = pickle.load(f)

    for mode in ['train', 'test']:
        for fold in range(10):
            print(mode, fold)
            graph_list, path_nx_list, align_list = [], [], []
            with open(os.path.join(data_path, 'sent_sample', f'10fold/{mode}_{fold}.pkl'), 'rb') as f:
                samples = pickle.load(f)
            for item in samples:
                sample_text = " ".join(item[1])
                sample_idx = text_dict[sample_text]
                sample_triple = amr_triple[sample_idx]
                sample_align = align_data[sample_idx]

                graph_list.append(load_amr_graphs(ent2id, rel2id, sample_triple))
                path_nx_list.append(build_path_graph(rel2id, sample_triple))
                align_list.append(sample_align)

            print('Save dgl graphs')
            with open(os.path.join(data_path, 'amr_sample/amr_graph', f'10fold/{mode}_{fold}.pkl'), "wb") as f:
                pickle.dump(graph_list, f)
            print('Save path networks')
            with open(os.path.join(data_path, 'amr_sample/amr_path', f'10fold/{mode}_{fold}.pkl'), "wb") as f:
                pickle.dump(path_nx_list, f)
            print('Save align data')
            with open(os.path.join(data_path, 'amr_sample/amr_align', f'10fold/{mode}_{fold}.pkl'), "wb") as f:
                pickle.dump(align_list, f)
    print('finish')

