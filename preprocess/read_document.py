import os
import bs4
import spacy
import pickle

from bs4 import BeautifulSoup as Soup
from collections import defaultdict
from typing import Dict, List, Set

nlp = spacy.load("en_core_web_sm")


def find_sent_id(sentences: List[Dict], mention_span: List[int]):
    """
    Find sentence id of mention
    """
    for sent in sentences:
        token_span_doc = sent['token_span_doc']
        if set(mention_span) == set(mention_span).intersection(set(token_span_doc)):
            return sent['sent_id']

    return None


def ctb_cat_reader(dir_name, file_name):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')

    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it.".format(dir_name + file_name))
        print(e)
        return None

    current_sent_id = -1
    tok_sent_id = -1
    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['id'])
        sent_id = int(tok.attrs['sentence'])
        # tok_sent_id = int(tok.attrs['number'])

        if sent_id == current_sent_id:
            tok_sent_id += 1
        else:
            current_sent_id = sent_id
            tok_sent_id = 0

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }

        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1

    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id - 1]))
        if start != 0:
            start = start + 1  # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['token_span_doc'] = _sent_token_span_doc[k]  # from 1
        sent_dict['content'] = ' '.join(v)
        sent_dict['tokens'] = v
        sent_dict['pos'] = []
        for tok in v:
            sent_dict['pos'].append(nlp(tok)[0].pos_)
        sent_dict['d_span'] = (start, start + len(sent_dict['content']))
        assert my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]] == sent_dict[
            'content'], f"\n'{sent_dict['content']}' \n '{my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]]}'"
        my_dict['sentences'].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(my_dict['doc_id'])
        return None

    for item in xml_dom.find('markables').children:
        if type(item) == bs4.element.Tag and 'event' in item.name:
            eid = int(item.attrs['id'])
            e_typ_list = item.attrs['class']
            if len(e_typ_list) == 1:
                e_typ = e_typ_list[0]
            else:
                print('error !!!')
                print(e_typ_list)

            mention_span = [int(anchor.attrs['id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]

            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0] - 1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0] - 1]))
                if start != 0:
                    start = start + 1  # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['d_span'] = (start, start + len(mention))
                my_dict['event_dict'][eid]['token_id_list'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = find_sent_id(my_dict['sentences'], mention_span)

                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(
                    mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"

    my_dict['relation_dict'] = {}
    for item in xml_dom.find('relations').children:
        if type(item) == bs4.element.Tag and 'clink' in item.name:
            r_id = item.attrs['id']
            r_typ = 'Cause'
            head = int(item.find('source').attrs['id'])
            tail = int(item.find('target').attrs['id'])

            assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
            my_dict['relation_dict'][(head, tail)] = r_typ

    return my_dict


if __name__ == '__main__':
    project_path = os.path.abspath('..')
    print(project_path)
    corpus_dir = os.path.join(project_path, 'data/raw_data/Causal-TimeBank/Causal-TimeBank-CAT/')
    data_list = []
    for index, file in enumerate(os.listdir(corpus_dir)):
        print(index, file)
        data_dict = ctb_cat_reader(corpus_dir, file)
        data_list.append(data_dict)

    data_dir = os.path.join(project_path, 'data/Causal-TimeBank')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    with open(os.path.join(project_path, 'data/Causal-TimeBank/document_raw_ctb.pkl'), 'wb') as f:
        pickle.dump(data_list, f)

    print('finish')
