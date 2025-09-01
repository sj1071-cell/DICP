import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, BertTokenizer


class BertEvent(nn.Module):

    def __init__(self, args):
        super(BertEvent, self).__init__()
        ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = BertTokenizer.from_pretrained(args.plm_path)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        self.config = BertConfig.from_pretrained(args.plm_path)
        self.bert = BertModel.from_pretrained(args.plm_path, config=self.config)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.args = args
        self.hidden_size = 768
    def forward(self, sentences_s, mask_s):
        # model forward propagation
        outputs = self.bert(sentences_s, attention_mask=mask_s)
        enc_s = outputs.last_hidden_state

        return enc_s
