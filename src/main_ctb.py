import os
import random
import dgl
import numpy as np
import torch

from transformers import BertTokenizer
from arguments_ctb import get_parser
from data_loader import Dataset, load_dict, load_pkl
from framework import Framework
from models.bert_event import BertEvent
from models.gcn_event import GCNEvent
from models.prompt4 import MLP
from amr_data_loader import Dataset_AMR
from utils.sampling import negative_sampling, positive_sampling
from utils.logger import get_logger

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)

    dgl.seed(seed)
    dgl.random.seed(seed)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    seed = args.random_seed
    set_seed(seed)
    logger = get_logger()
    logger.info("Set seed: {}".format(seed))

    args.save_model_name = f"{args.save_model}_MLP_fold_{args.fold}.pt"
    args.save_model_name2 = f"{args.save_model}_bert_init_fold_{args.fold}.pt"
    args.save_model_name3 = f"{args.save_model}_gcn_fold_{args.fold}.pt"
    args.board_dir = f'output/tensorboard_log/{args.save_model}_fold_{args.fold}'

    # get current path
    file_dir = os.path.dirname(__file__)
    args.project_path = os.path.abspath(os.path.join(file_dir, '..'))
    logger.info("project_path: {}".format(args.project_path))

    # create save_model_dir
    # for file_dir in [args.save_model_dir, args.output_dir]:
    #     file_dir = os.path.join(args.project_path, file_dir)
    #     if not os.path.exists(file_dir):
    #         os.mkdir(file_dir)
    file_dir = os.path.join(args.project_path, args.output_dir)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    file_dir = os.path.join('/mnt/disk2/sj', args.save_model_dir)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    args.save_model_path = os.path.join('/mnt/disk2/sj', args.save_model_dir)
    logger.info("save_model_path: {}".format(args.save_model_path))

    # set device
    args.device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    logger.info("Device: {}".format(args.device))

    logger.info("Load data from fold {}".format(args.fold))
    # load sample data
    train_file = os.path.join(args.project_path, args.sample_dir, '10fold', f'train_{args.fold}.pkl')
    test_file = os.path.join(args.project_path, args.sample_dir, '10fold', f'test_{args.fold}.pkl')
    train_set = load_pkl(train_file)
    test_set = load_pkl(test_file)

    # read ent and rel dict
    ent2id = load_dict(os.path.join(args.project_path, args.vertex_dict))
    rel2id = load_dict(os.path.join(args.project_path, args.edge_dict))
    args.num_ents = len(ent2id.keys())
    args.num_rels = len(rel2id.keys())

    # load amr graphs
    graph_train = os.path.join(args.project_path, args.graph_dir, '10fold', f'train_{args.fold}.pkl')
    graph_test = os.path.join(args.project_path, args.graph_dir, '10fold', f'test_{args.fold}.pkl')
    train_graphs = load_pkl(graph_train)
    test_graphs = load_pkl(graph_test)

    # load align info
    align_train = os.path.join(args.project_path, args.align_dir, '10fold', f'train_{args.fold}.pkl')
    align_test = os.path.join(args.project_path, args.align_dir, '10fold', f'test_{args.fold}.pkl')
    train_align_info = load_pkl(align_train)
    test_align_info = load_pkl(align_test)

    # load path graph
    path_train = os.path.join(args.project_path, args.path_dir, '10fold', f'train_{args.fold}.pkl')
    path_test = os.path.join(args.project_path, args.path_dir, '10fold', f'test_{args.fold}.pkl')
    train_data_path = load_pkl(path_train)
    test_data_path = load_pkl(path_test)

    assert (len(train_set) == len(train_graphs))
    assert (len(train_set) == len(train_align_info))
    assert (len(train_set) == len(train_data_path))

    if args.negative_sample:
        train_set, train_graphs, train_align_info, train_data_path = \
            negative_sampling(train_set, train_graphs, train_align_info, train_data_path, ratio=args.sample_ratio)
    if args.positive_sample:
        train_set, train_graphs, train_align_info, train_data_path = \
            positive_sampling(train_set, train_graphs, train_align_info, train_data_path, ratio=args.pos_ratio)

    logger.info("The number of train set is {}".format(len(train_set)))
    logger.info("The number of test set is {}".format(len(test_set)))

    tokenizer = BertTokenizer.from_pretrained(args.plm_path)

    # train_indices, train_sentences, train_events, train_labels = find_train_Demo(args, train_set, tokenizer)
    # test_indices = find_test_Demo(args, train_set, test_set)

    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>", "<t1>", "</t1>", "</na>", "</causal>"]
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    args.vocab_size = len(tokenizer)

    model_init = BertEvent(args).to(args.device)

    model_gcn = GCNEvent(args).to(args.device)

    answer_space = [30528, 30529]  # </na>, </causal>

    # set model
    logger.info("Model is {}.".format(args.model))

    # 该模型是用于对提示中的mask进行预测
    MLP = MLP(args, tokenizer).to(args.device)

    # create training framework
    framework = Framework(args)
    logger.info(args)

    logger.info("Loading test dataset ...")
    test_dataset = Dataset(args, test_set, tokenizer)
    test_dataset_batch = [batch for batch in test_dataset.reader(args.device)]
    test_dataset2 = Dataset_AMR(args, test_set, test_graphs, test_align_info, test_data_path, tokenizer)
    test_dataset_batch2 = [batch for batch in test_dataset2.reader(args.device)]

    if not args.only_test:
        logger.info("Loading train dataset ...")
        train_dataset = Dataset(args, train_set, tokenizer)
        train_dataset2 = Dataset_AMR(args, train_set, train_graphs, train_align_info, train_data_path, tokenizer)
        framework.train(train_dataset, train_dataset2, test_dataset_batch, test_dataset_batch2, MLP,
                        model_init, model_gcn, args.device, answer_space)

    logger.info("Loading best model ...")
    MLP.load_state_dict(
        torch.load(os.path.join(args.save_model_path, args.save_model_name), map_location=args.device))
    model_init.load_state_dict(
        torch.load(os.path.join(args.save_model_path, args.save_model_name2), map_location=args.device))
    model_gcn.load_state_dict(
        torch.load(os.path.join(args.save_model_path, args.save_model_name3), map_location=args.device))
    precision, recall, f1 = framework.evaluate(test_dataset_batch, test_dataset_batch2, MLP, model_init,
                                               model_gcn, args.device, answer_space)
    logger.info("Precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n".format(precision, recall, f1))