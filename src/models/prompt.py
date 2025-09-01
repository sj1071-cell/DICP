# coding: UTF-8

import torch
import torch.nn as nn
from torch.nn.functional import gelu
from transformers import RobertaForMaskedLM, BertForMaskedLM

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, vocab_size, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.decoder = nn.Linear(output_dim, vocab_size)
        self.tanh = nn.Tanh()
        self.layer_norm = nn.LayerNorm(output_dim, eps=1e-12)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear(x)
        if self.use_activation:
            x = self.tanh(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class MLP(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.device = args.device
        self.BERT_MLM = BertForMaskedLM.from_pretrained(args.plm_path).to(self.device)
        self.BERT_MLM.resize_token_embeddings(args.vocab_size)
        self.num_layers = args.num_layers
        self.hidden_size = 768
        self.args = args

        self.tokenizer = tokenizer
        self.vocab_size = args.vocab_size
        self.decoder_hidden_size = args.decoder_hidden_size
        self.lm_head = FCLayer(
            self.decoder_hidden_size,
            self.decoder_hidden_size,
            self.vocab_size,
            use_activation=True,
        )

        for param in self.BERT_MLM.parameters():
            param.requires_grad = True

        # self.temperature = args.temperature

    def get_embedding(self, sentences_s):
        # 得到前缀初始化表示
        embeddings = []
        prefix_mask = []

        # 使用 zip() 合并每个列表中相同位置的元素
        combined = [list(pair) for pair in zip(*sentences_s)]
        for index, sentence_s in enumerate(combined):
            encoding = self.tokenizer(
                sentence_s,
                padding="max_length",
                max_length=self.args.len_arg,
                truncation=True,
                return_tensors="pt"
            )

            # 对句子进行编码并填充
            attention_mask = encoding['attention_mask'].to(self.device)

            # 获取模型的词嵌入权重
            word_embeddings = self.BERT_MLM.bert.embeddings.word_embeddings

            # 使用 token IDs 通过词嵌入层来获取嵌入向量
            input_ids = encoding['input_ids'].to(self.device)

            # 通过词嵌入获取嵌入向量
            embedding = word_embeddings(input_ids).to(self.device)
            embeddings.append(embedding)
            # embeddings[index] = embedding
            # prefix_mask[index] = attention_mask
            prefix_mask.append(attention_mask)

        # 把词嵌入层和前缀向量长度返回

        return embeddings, prefix_mask

    def forward(self, arg, mask_arg, token_mask_indices, demons, prefix_mask):
        batch_size = len(arg)

        out_arg = self.BERT_MLM.bert(arg, attention_mask=mask_arg, output_hidden_states=True)[0]

        input_with_prefix = out_arg
        extended_mask_arg = mask_arg

        # 通过模型每一层
        all_hidden_states = []
        for layer_index, layer in enumerate(self.BERT_MLM.bert.encoder.layer):
            if (layer_index < self.args.demon_num):
                input_with_prefix = torch.cat([demons[layer_index], input_with_prefix], dim=1)
                # 将前缀mask与其拼接
                extended_mask_arg = torch.cat([prefix_mask[layer_index], extended_mask_arg], dim=1)
                extended_mask_arg4 = extended_mask_arg[:, None, None, :]
                input_with_prefix = layer(input_with_prefix, attention_mask=extended_mask_arg4)[0]
                all_hidden_states.append(input_with_prefix)
            else:
                input_with_prefix = layer(input_with_prefix, attention_mask=extended_mask_arg4)[0]
                all_hidden_states.append(input_with_prefix)

        # 获取最终层的输出
        final_output = all_hidden_states[-1]

        # 获取目标词的隐藏状态
        anchor_hidden_mask = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        for i in range(batch_size):
            anchor_hidden_mask[i] = final_output[i][token_mask_indices[i] + self.args.len_arg * (self.args.demon_num)]

        # 通过语言模型头得到输出词表
        # out_vocab = self.RoBERTa_MLM.lm_head(anchor_hidden_mask)
        # out_ans = out_vocab[:, Answer_id]

        return anchor_hidden_mask

