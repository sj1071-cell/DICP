import os
import argparse
import penman
import json
import spacy
import pickle

from penman.models import amr, noop
from spacy.tokens import Doc
from amrlib.alignments.rbw_aligner import RBWAligner

_exclude_components = ["parser"]
TOKENIZER = spacy.load("en_core_web_sm", exclude=_exclude_components)


def add_lemmas(graph):
    """Add tokens and lemmas to penman graph."""
    # Spacy may produce different result if snt is tokenized.
    # For example, "I 'm" will be tokenized into "I ' m".
    # So, do not use add_lemmas in amrlib if your snt is tokenized!!!
    penman_graph = penman.decode(graph, model=amr.model)
    snt = penman_graph.metadata["snt"]
    penman_graph.metadata["tokens"] = json.dumps(snt.split())
    # Use spacy to get lemmas
    doc = Doc(TOKENIZER.vocab, words=snt.split())
    for name, proc in TOKENIZER.pipeline:
        doc = proc(doc)
    lemmas = []
    for t in doc:
        if t.lemma_ == "-PRON-":
            lemma = t.text.lower()
        elif t.tag_.startswith("NNP") or t.ent_type_ not in ("", "O"):
            lemma = t.text
        else:
            lemma = t.lemma_.lower()
        lemmas.append(lemma)
    # Add lemma
    penman_graph.metadata["lemmas"] = json.dumps(lemmas)
    return penman_graph


def align_graph(graph):
    """Align single amr graph."""
    # penman_graph = add_lemmas(graph, snt_key='snt')
    penman_graph = add_lemmas(graph)
    align_result = RBWAligner.from_penman_w_json(penman_graph)
    ret_val = []
    # Return in one line, "<index0> <short0>\t<index1> <short1>\t..."
    for i, t in enumerate(align_result.alignments):
        if t is not None:
            # t.triple: (short, ":instance", name)
            ret_val.append((i, t.triple[0], t.triple[1], t.triple[2]))
    # ret_val = "\t".join(["{} {}".format(i, s) for i, s in ret_val])
    return ret_val


def map_align_data(vertex_dict, amr_text):
    res = align_graph(amr_text)
    penman_graph = penman.decode(amr_text, model=amr.model)

    # instances in one sentence have same label
    graph_node_dict = {}
    for _id, _, instance in penman_graph.instances():
        if instance not in graph_node_dict.values():
            instance_label = instance
        else:
            instance_label = f'{instance}__{_id}'
        graph_node_dict[_id] = instance_label

    # edges
    graph_edge_dict = {}
    for head, relation, tail in penman_graph.edges():
        graph_edge_dict[head + ' ' + relation] = tail

    # attributes
    graph_att_dict = {}
    for _sid, edge, attribute in penman_graph.attributes():
        if attribute not in graph_att_dict.values():
            attribute_babel = attribute
        else:
            attribute_babel = f'{attribute}__{_sid}'
        graph_att_dict[_sid + ' ' + edge] = attribute_babel

    align_dict_new = {}
    for item in res:
        # build triple, graph_node_dict -> node_id: label
        if item[2] == ':instance':
            align_dict_new[int(item[0])] = int(vertex_dict[graph_node_dict[item[1]]])
        elif item[1] + ' ' + item[2] in graph_edge_dict.keys():
            _id = graph_edge_dict[item[1] + ' ' + item[2]]
            align_dict_new[int(item[0])] = int(vertex_dict[graph_node_dict[_id]])
        elif item[1] + ' ' + item[2] in graph_att_dict.keys():
            _att = graph_att_dict[item[1] + ' ' + item[2]]
            align_dict_new[int(item[0])] = int(vertex_dict[_att])

    return align_dict_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Causal-TimeBank")
    args = parser.parse_args()

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data', args.dataset_name)

    dir_list = ['amr_sample/amr_align']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    with open(os.path.join(data_path, 'amr_data/sentences.txt'), 'r') as f:
        res = f.read()
    text_list = res.split('\n\n')

    print('generate align data')
    with open(os.path.join(data_path, 'amr_sample/amr_dict/vertex_dict.txt'), 'r') as f:
        vertex_dict = {line.rstrip('\n'): index for index, line in enumerate(f.readlines())}

    align_data = []
    for amr_text in text_list:
        amr_align_item = map_align_data(vertex_dict, amr_text)
        align_data.append(amr_align_item)

    with open(os.path.join(data_path, 'amr_sample/amr_align', 'align_data.pkl'), 'wb') as f:
        pickle.dump(align_data, f)

    print('finish')
