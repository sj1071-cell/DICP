import os
import pickle
import random

from sklearn.model_selection import KFold


if __name__ == '__main__':
    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data/Causal-TimeBank')

    dir_list = ['sent_sample/10fold']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    with open(os.path.join(data_path, 'data_samples.pkl'), 'rb') as f:
        documents = pickle.load(f)

    index = list(documents.keys())
    random.seed(6688)
    random.shuffle(index)
    kfold = KFold(n_splits=10)

    for fold, (train_doc_ids, dev_doc_ids) in enumerate(kfold.split(index)):
        print('fold', fold)

        train_set = []
        dev_set = []
        for train_id in train_doc_ids:
            train_set.extend(documents[index[train_id]])
        for dev_id in dev_doc_ids:
            dev_set.extend(documents[index[dev_id]])

        random.shuffle(train_set)
        with open(os.path.join(data_path, 'sent_sample/10fold', f'train_{fold}.pkl'),
                  'wb') as f:
            pickle.dump(train_set, f)
        with open(os.path.join(data_path, 'sent_sample/10fold', f'test_{fold}.pkl'), 'wb') as f:
            pickle.dump(dev_set, f)

    print('finish')