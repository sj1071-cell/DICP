import os
import argparse
import penman
from penman.models import amr, noop


def get_graph_triple(data_path, reverse=True):
    amr_dict_path = os.path.join(data_path, 'amr_sample/amr_dict')
    amr_sent_path = os.path.join(data_path, 'amr_data')
    amr_triple_path = os.path.join(data_path, 'amr_sample/amr_triple')

    # create amr graph dir
    if not os.path.exists(amr_triple_path):
        os.makedirs(amr_triple_path)

    # load amr dict
    with open(os.path.join(amr_dict_path, 'vertex_dict.txt'), 'r') as f:
        vertex_dict = {line.rstrip('\n'): index for index, line in enumerate(f.readlines())}
    with open(os.path.join(amr_dict_path, 'edge_dict.txt'), 'r') as f:
        edge_dict = {line.rstrip('\n'): index for index, line in enumerate(f.readlines())}

    # output graph triple
    file_path = os.path.join(amr_sent_path, 'sentences.txt')
    with open(file_path, 'r') as f:
        res = f.read()
    text_list = res.split('\n\n')

    triple_all = []
    for index, amr_text in enumerate(text_list):
        if reverse:
            model = amr.model
        else:
            model = noop.model
        penman_graph = penman.decode(amr_text, model=model)

        # text = "In the days following Gray's death , it's been revealed that both Mourad and Cordova have reportedly"
        # if amr_text.find(text) > -1:
        #     print("text:", index)

        # build triple, graph_node_dict -> node_id: label
        graph_node_list, graph_att_list = [], []
        graph_node_dict, graph_att_dict = {}, {}

        # instances in one sentence have same label
        for _id, _, instance in penman_graph.instances():
            if instance not in graph_node_list:
                graph_node_list.append(instance)
                graph_node_dict[_id] = instance
            else:
                instance_label = f'{instance}__{_id}'
                graph_node_list.append(instance_label)
                graph_node_dict[_id] = instance_label

        # build edges
        triple_list = []
        for head, relation, tail in penman_graph.edges():
            head_id = vertex_dict[graph_node_dict[head]]
            tail_id = vertex_dict[graph_node_dict[tail]]
            relation_id = edge_dict[relation]
            triple_list.append((head_id, relation_id, tail_id))

        for _sid, edge, attribute in penman_graph.attributes():
            if attribute not in graph_att_dict.values():
                attribute_babel = attribute
            else:
                attribute_babel = f'{attribute}__{_sid}'
            graph_att_dict[_sid + ' ' + edge] = attribute_babel

        # build attributes
        for head, relation, tail in penman_graph.attributes():
            head_id = vertex_dict[graph_node_dict[head]]
            tail_id = vertex_dict[graph_att_dict[head + ' ' + relation]]
            relation_id = edge_dict[relation]
            triple_list.append((head_id, relation_id, tail_id))

        triple_list.sort()
        triple_all.append(triple_list)

    with open(os.path.join(amr_triple_path, f'amr_triple.txt'), 'w') as f:
        for triple_sample in triple_all:
            for item in triple_sample:
                f.write(str(item) + '\n')
            f.write('\n\n')
        print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Causal-TimeBank")
    args = parser.parse_args()

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data', args.dataset_name)

    get_graph_triple(data_path)