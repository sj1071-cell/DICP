import penman
import os
import argparse
from penman.models import amr, noop


def get_amr_dict(data_path, reverse=True):
    """
    get all data amr dict.
    :param data_path: data path.
    :param reverse: if '-of'-like relations are reversed.
    :return: vertex_dict, edge_dict, sent_dict.
    """

    vertex_set = set()
    edge_set = set()
    sent_set = set()

    file_path = os.path.join(data_path, 'amr_data/sentences.txt')
    with open(file_path, 'r') as f:
        res = f.read()
    text_list = res.split('\n\n')

    for amr_text in text_list:
        if reverse:
            model = amr.model
        else:
            model = noop.model
        penman_graph = penman.decode(amr_text, model=model)

        # duplicate triple exists in amr graph
        # error_ex = penman_graph._filter_triples(source='s', role=':ARG0', target='p3')
        # if len(error_ex) > 1:
        #     print(penman_graph)
        #     pass

        graph_node_list, graph_att_list = [], []

        sent_set.add(penman_graph.metadata['snt'])
        for _, edge, _ in penman_graph.edges():
            edge_set.add(edge)

        # instances in one sentence have same label
        for _id, _, instance in penman_graph.instances():
            if instance not in graph_node_list:
                graph_node_list.append(instance)
            else:
                graph_node_list.append(f'{instance}__{_id}')

        # attributes in one sentence have same label
        for _sid, edge, attribute in penman_graph.attributes():
            edge_set.add(edge)
            if attribute not in graph_att_list:
                graph_att_list.append(attribute)
            else:
                graph_att_list.append(f'{attribute}__{_sid}')

        vertex_set = vertex_set.union(set(graph_node_list), set(graph_att_list))

    vertex_list, edge_list, sent_list = list(vertex_set), list(edge_set), list(sent_set)
    vertex_list.sort(), edge_list.sort(), sent_list.sort()

    return vertex_list, edge_list, sent_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Causal-TimeBank")
    args = parser.parse_args()

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data', args.dataset_name)

    dir_list = ['amr_sample/amr_dict']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    vertex_list, edge_list, sent_list = get_amr_dict(data_path, reverse=True)

    with open(os.path.join(data_path, 'amr_sample/amr_dict/vertex_dict.txt'), 'w') as f:
        for item in vertex_list:
            f.write(item + '\n')
    with open(os.path.join(data_path, 'amr_sample/amr_dict/edge_dict.txt'), 'w') as f:
        for item in edge_list:
            f.write(item + '\n')

    print("finish")