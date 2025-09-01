import random
import pickle

from collections import Counter
from .logger import get_logger


def negative_sampling(train_set, train_graphs, train_align_info, train_data_dfs, ratio=0.5):
    logger = get_logger()

    res_a, res_b, res_c, res_d = [], [], [], []
    label_before = []
    label_after = []
    for a, b, c, d in zip(train_set, train_graphs, train_align_info, train_data_dfs):
        label_before.append(a[4])
        if a[4] == 'NULL':
            if random.random() < ratio:
                continue
        label_after.append(a[4])
        res_a.append(a)
        res_b.append(b)
        res_c.append(c)
        res_d.append(d)

    label_before_cnt = Counter(label_before)
    label_after_cnt = Counter(label_after)
    pos_num = label_before_cnt['FALLING_ACTION'] + label_before_cnt['PRECONDITION'] + label_before_cnt['Cause'] + \
              label_before_cnt['null'] + label_before_cnt['plot_null']
    neg_num = label_before_cnt['NULL']
    logger.info('*** before negative sample:')
    logger.info(f'total number is {len(label_before)}')
    logger.info(f"pos label num is {pos_num}")
    logger.info(f"neg label num is {neg_num}")

    pos_num = label_after_cnt['FALLING_ACTION'] + label_after_cnt['PRECONDITION'] + label_after_cnt['Cause'] + \
              label_after_cnt['null'] + label_before_cnt['plot_null']
    neg_num = label_after_cnt['NULL']
    logger.info('*** after negative sample:')
    logger.info(f'total number is {len(label_after)}')
    logger.info(f"pos label num is {pos_num}")
    logger.info(f"neg label num is {neg_num}")

    return res_a, res_b, res_c, res_d


def positive_sampling(train_set, train_graphs, train_align_info, train_data_dfs, ratio=3):
    logger = get_logger()

    res_a, res_b, res_c, res_d = [], [], [], []
    label_before = []
    label_after = []
    for a, b, c, d in zip(train_set, train_graphs, train_align_info, train_data_dfs):
        label_before.append(a[4])
        if a[4] != 'NULL':
            for i in range(ratio):
                label_after.append(a[4])
                res_a.append(a)
                res_b.append(b)
                res_c.append(c)
                res_d.append(d)
        else:
            label_after.append(a[4])
            res_a.append(a)
            res_b.append(b)
            res_c.append(c)
            res_d.append(d)

    label_before_cnt = Counter(label_before)
    label_after_cnt = Counter(label_after)
    pos_num = label_before_cnt['FALLING_ACTION'] + label_before_cnt['PRECONDITION'] + label_before_cnt['Cause'] + \
              label_before_cnt['null'] + label_before_cnt['plot_null']
    neg_num = label_before_cnt['NULL']
    logger.info('*** before positive sample:')
    logger.info(f'total number is {len(label_before)}')
    logger.info(f"pos label num is {pos_num}")
    logger.info(f"neg label num is {neg_num}")

    pos_num = label_after_cnt['FALLING_ACTION'] + label_after_cnt['PRECONDITION'] + label_after_cnt['Cause'] + \
              label_after_cnt['null'] + label_before_cnt['plot_null']
    neg_num = label_after_cnt['NULL']
    logger.info('*** after positive sample:')
    logger.info(f'total number is {len(label_after)}')
    logger.info(f"pos label num is {pos_num}")
    logger.info(f"neg label num is {neg_num}")

    return res_a, res_b, res_c, res_d
