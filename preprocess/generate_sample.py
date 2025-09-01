import os
import pickle
import itertools


def get_sent(dataset):
    sent_list = []
    for item in dataset:
        sent = ' '.join(item[1])
        sent_list.append(sent)

    return sent_list


if __name__ == '__main__':
    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data/Causal-TimeBank')

    with open(os.path.join(data_path, 'document_raw_ctb.pkl'), 'rb') as f:
        documents = pickle.load(f)

    data_set = {}
    index = 0
    event_num = 0
    clink_num = 0
    pos_num = 0
    for doc_dict in documents:
        doc_name = doc_dict['doc_id']
        print('doc', doc_name)

        events_list = doc_dict['event_dict'].keys()
        event_combination = itertools.combinations(events_list, 2)  # 单向
        event_num += len(events_list)

        clink_num += len(doc_dict['relation_dict'])

        doc_data = []
        for item in event_combination:
            event1 = item[0]
            event2 = item[1]
            if event1 == event2:  # event ID
                continue

            # Causal Relation
            rel = 'NULL'
            if (event1, event2) in doc_dict['relation_dict'].keys():
                rel = doc_dict['relation_dict'][(event1, event2)]
            elif (event2, event1) in doc_dict['relation_dict'].keys():
                rel = doc_dict['relation_dict'][(event2, event1)]

            sen_s = doc_dict['event_dict'][event1]['sent_id']
            sen_t = doc_dict['event_dict'][event2]['sent_id']

            # same sentence
            if sen_s == sen_t:
                sentence_s = doc_dict['sentences'][sen_s]['tokens']

                span1 = doc_dict['event_dict'][event1]['token_id_list']
                span2 = doc_dict['event_dict'][event2]['token_id_list']

                doc_data.append([index, sentence_s, span1, span2, rel])
                index += 1
                if rel != 'NULL':
                    pos_num += 1

        data_set[doc_name] = doc_data

    dir_list = ['sent_sample']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    print('doc num:', len(documents))  # 183
    print('event num:', event_num)  # 6811
    print('item num:', index)  # 9721
    print('clink num:', clink_num)  # 318
    print('pos num:', pos_num)  # 298

    with open(os.path.join(data_path, 'data_samples.pkl'), 'wb') as f:
        pickle.dump(data_set, f)

    train_dev_data =[]
    for doc_name in data_set.keys():
        train_dev_data.extend(data_set[doc_name])

    with open(os.path.join(data_path, 'sent_sample/train_dev.pkl'), 'wb') as f:
        pickle.dump(train_dev_data, f)
    print('train dev data num:', len(train_dev_data))

    print('finish')