# @Author : bamtercelboo
# @Datetime : 2018/8/16 8:50
# @File : Instance_extend.py
# @Last Modify Time : 2018/8/16 8:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Instance_extend.py
    FUNCTION : None
"""
from .Instance_Base import Instance_Base
import torch
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance(Instance_Base):
    def __init__(self):
        # super(Instance).__init__()
        # print("Instance extend")
        super().__init__()
        self.fact = []
        self.accu_labels = []
        self.law_labels = []
        self.prison_labels = []
        self.prison_gold_labels = []
        self.topk_accu = []
        self.law_article = []
        self.accusation = []
        self.law_art_accu = []

        self.accu_labels_size = 0
        self.law_labels_size = 0
        self.prison_labels_size = 0
        self.topk_accu_size = 0
        self.sentence_size = 0
        self.next_sentence_size = 0
        self.sentence_word_size = []
        self.accusation_size = 0
        self.law_art_accu_size = 0

        self.accu_label_index = []
        self.law_label_index = []
        self.prison_label_index = []
        self.topk_accu_index = []
        self.law_article_index = []
        self.accusation_index = []
        self.law_art_accu_index = []


