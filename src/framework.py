import os
import torch
import torch.nn as nn
import itertools

from tqdm import tqdm
from transformers import AdamW
from sklearn import metrics
import torch.nn.functional as F

from utils.logger import get_logger
class Framework:

    def __init__(self, args):
        self.args = args
        self.logger = get_logger()

    def train_step(self, batch, batch2, MLP, model_init, model_gcn, device, answer_space):
        # train step in stage of training
        model_gcn.train()
        model_init.train()
        MLP.train()

        # bert进行预测
        input_ids, masks, mask_indices, data_y, batch_demons = batch
        input_ids = input_ids.to(device)
        masks = masks.to(device)
        mask_indices = mask_indices.to(device)

        demons_encoder, prefix_mask = MLP.get_embedding(batch_demons)
        prefix = MLP.forward(input_ids, masks, mask_indices, demons_encoder, prefix_mask)

        # 得到AMR信息
        sentences_s, mask_s, event1, event1_mask, event2, event2_mask, graphs, graphs_len, \
            graph_event1, graph_event1_mask, graph_event2, graph_event2_mask, data_y, \
            paths_list, paths_len_list, bert_init_list, random_init_list = batch2

        # get bert representations
        sent_h = model_init.forward(sentences_s, mask_s)

        # get graph representations
        gcn_opt = model_gcn.forward(graphs, graphs_len, graph_event1, graph_event1_mask,
                                    graph_event2, graph_event2_mask, paths_list, paths_len_list,
                                    bert_init_list, random_init_list, sent_h)

        ensemble_opt = torch.cat([prefix, gcn_opt], dim=1)
        # 将最终向量表示放入到MLM Head中
        out_vocab = MLP.lm_head(ensemble_opt)
        pre = out_vocab[:, answer_space]

        # Answer space：[30528, 30529]
        predt = torch.argmax(pre, dim=1).detach()

        # calculate loss
        loss = F.cross_entropy(pre, data_y)
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_gcn.parameters(), self.args.max_grad_norm)

        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()

        return loss

    def train(self, train_dataset, train_dataset2, dev_dataset_batch, dev_dataset_batch2, MLP, model_init, model_gcn, device, answer_space):
        # get optimizer schedule
        self.optimizer = AdamW(itertools.chain(MLP.parameters(), model_init.parameters()),
                               lr=self.args.bert_learning_rate)
        self.optimizer2 = AdamW(model_gcn.parameters(), lr=self.args.rgcn_learning_rate)
        # Train
        self.logger.info("***** Running training *****")
        self.logger.info(
            "Learning rate of BERT: {:.8f}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.logger.info(
            "Learning rate of GCN: {:.8f}".format(self.optimizer2.state_dict()['param_groups'][0]['lr']))

        best_result = 0

        train_dataset_batch = [batch for batch in
                               train_dataset.reader(self.args.device, shuffle=self.args.epoch_shuffle)]
        train_dataset_batch2 = [batch for batch in
                               train_dataset2.reader(self.args.device, shuffle=self.args.epoch_shuffle)]
        # self.logger.debug('train dataset: {}'.format(dict(sorted(Counter(train_dataset.paths_len_list).items()))))
        for epoch in range(0, int(self.args.num_train_epochs)):
            self.logger.info("***** Training *****")
            self.logger.info("Epoch: {}/{}".format(epoch + 1, self.args.num_train_epochs))

            if self.args.epoch_shuffle and epoch != 0:
                train_dataset_batch = [batch for batch in
                                       train_dataset.reader(self.args.device, shuffle=self.args.epoch_shuffle)]
            loss_train = 0
            for step, (batch1,batch2) in enumerate(tqdm(zip(train_dataset_batch,train_dataset_batch2), ncols=80, mininterval=10)):
            # for step, batch in enumerate(tqdm(train_dataset_batch, ncols=80, mininterval=10)):
                loss = self.train_step(batch1, batch2, MLP, model_init, model_gcn, device, answer_space)
                loss_train += loss
            self.logger.info("Train loss: {:.6f}.".format(loss_train / len(train_dataset_batch)))

            # evaluate
            # precision, recall, f1 = self.evaluate(dev_dataset_batch, model_bert, device, answer_space)
            precision, recall, f1 = self.evaluate(dev_dataset_batch, dev_dataset_batch2, MLP, model_init, model_gcn, device,
                                                  answer_space)
            self.logger.info("Precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(precision, recall, f1))
            self.logger.info("Best f1: {:.3f}, current f1: {:.3f}.".format(best_result, f1))

            if f1 > best_result:
                best_result = f1
                self.logger.info(
                    "New best f1: {:.3f}. Saving model checkpoint.".format(best_result))
                torch.save(MLP.state_dict(), os.path.join(self.args.save_model_path, self.args.save_model_name))
                torch.save(model_init.state_dict(), os.path.join(self.args.save_model_path, self.args.save_model_name2))
                torch.save(model_gcn.state_dict(), os.path.join(self.args.save_model_path, self.args.save_model_name3))

    def evaluate(self, dataset_batch, dataset_batch2, MLP, model_init, model_gcn, device, answer_space):
        self.logger.info("***** Evaluating *****")

        model_gcn.eval()
        model_init.eval()
        MLP.eval()

        loss_eval = 0
        predicted_all = []
        gold_all = []

        with torch.no_grad():
            # for step, batch in enumerate(tqdm(dataset_batch, ncols=80, mininterval=5)):
            for step, (batch_data, batch_data2) in enumerate(tqdm(zip(dataset_batch, dataset_batch2), ncols=80, mininterval=5)):
                input_ids, masks, mask_indices, data_y, batch_demons = batch_data
                input_ids = input_ids.to(device)
                masks = masks.to(device)
                mask_indices = mask_indices.to(device)

                demons_encoder, prefix_mask = MLP.get_embedding(batch_demons)
                prefix = MLP.forward(input_ids, masks, mask_indices, demons_encoder, prefix_mask)

                # 得到AMR信息
                sentences_s, mask_s, event1, event1_mask, event2, event2_mask, graphs, graphs_len, \
                    graph_event1, graph_event1_mask, graph_event2, graph_event2_mask, data_y, \
                    paths_list, paths_len_list, bert_init_list, random_init_list = batch_data2

                # get bert representations
                sent_h = model_init.forward(sentences_s, mask_s)
                # _, sent_h = model_init.forward(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)

                # get graph representations
                gcn_opt = model_gcn.forward(graphs, graphs_len, graph_event1, graph_event1_mask,
                                            graph_event2, graph_event2_mask, paths_list, paths_len_list,
                                            bert_init_list, random_init_list, sent_h)

                ensemble_opt = torch.cat([prefix, gcn_opt], dim=1)
                # 将最终向量表示放入到MLM Head中
                out_vocab = MLP.lm_head(ensemble_opt)
                pre = out_vocab[:, answer_space]

                # calculate loss
                loss = F.cross_entropy(pre, data_y)

                predicted = torch.argmax(pre, dim=1).detach()
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                loss_eval += loss
                gold = list(data_y.cpu().numpy())
                gold_all += gold

                # calculate f1, precision, recall and acc
                f1 = metrics.f1_score(gold_all, predicted_all) * 100
                precision = metrics.precision_score(gold_all, predicted_all) * 100
                recall = metrics.recall_score(gold_all, predicted_all) * 100

        return precision, recall, f1