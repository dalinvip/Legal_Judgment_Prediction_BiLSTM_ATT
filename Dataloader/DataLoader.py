# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import os
import sys
import re
import random
import json
import torch
import numpy as np
from collections import OrderedDict
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _clean_punctuation(string):
        """
        :param string:
        :return:
        """
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"，", "", string)
        string = re.sub(r"。", "", string)
        string = re.sub(r"“", "", string)
        string = re.sub(r"”", "", string)
        string = re.sub(r"、", "", string)
        string = re.sub(r"：", "", string)
        string = re.sub(r"；", "", string)
        string = re.sub(r"（", "", string)
        string = re.sub(r"）", "", string)
        string = re.sub(r"《 ", "", string)
        string = re.sub(r"》", "", string)
        # string = re.sub(r"× ×", "", string)
        # string = re.sub(r"x")
        string = re.sub(r"  ", " ", string)
        return string.lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts

    @staticmethod
    def _sortby_sencount(insts):
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.sentence_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=False)
        for key, value in dict:
            if len(sorted_insts) > 0:
                sorted_insts[-1].next_sentence_size = insts[key].sentence_size
            sorted_insts.append(insts[key])
        sorted_insts[-1].next_sentence_size = -1
        print("Sort by Doc Sentence Count Finished.")
        return sorted_insts

    @staticmethod
    def _sort2json_file(insts, path):
        path = "_".join([path, "sort.json"])
        print("Sort Result To File {}.".format(path))
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for inst in insts:
            dictObj = {'meta': {'accusation': inst.accu_labels, 'predict_accu': []}, 'fact': " ".join(inst.words)}
            jsObj = json.dumps(dictObj, ensure_ascii=False)
            file.write(jsObj + "\n")
        file.close()
        print("Sort Result To File Finished.")


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle
        self.max_train_len = config.max_train_len

    def dataLoader(self):
        """
        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_JsonData(path=path[id_data])
            if shuffle is True and id_data == 0:
                print("shuffle train data......")
                random.shuffle(insts)
            # if self.split_doc is True:
            #     sentence_insts = self._from_chapter2default_cut(insts, cutoff=self.word_cut_count)
            insts = self._sort(insts=insts)
            self._sort2json_file(insts, path=path[id_data])
            self.data_list.append(insts)
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_JsonData(self, path=None, train=False):
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0
        print()
        with open(path, encoding="UTF-8") as f:
            lines = f.readlines()
            for line in lines:
                now_lines += 1
                if now_lines % 2000 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                if line == "\n":
                    print("empty line")
                inst = Instance()
                line_json = json.loads(line)
                fact = line_json["fact"].split()[:self.max_train_len]

                # accu label
                accu = line_json["meta"]["accusation"]
                # print(accu)
                # law label
                law = line_json["meta"]["relevant_articles"]
                # prison label
                death_penalty = line_json["meta"]["term_of_imprisonment"]["death_penalty"]
                life_imprisonment = line_json["meta"]["term_of_imprisonment"]["life_imprisonment"]
                imprisonment = line_json["meta"]["term_of_imprisonment"]["imprisonment"]
                if death_penalty is True:
                    v = 0
                elif life_imprisonment is True:
                    v = 1
                else:
                    v = 2

                # inst.words.append(" ".join(fact))
                inst.words = fact
                inst.accu_labels = accu
                inst.law_labels = law
                inst.prison_labels = [v]

                if death_penalty is True:
                    inst.gold_labels = death
                elif life_imprisonment is True:
                    inst.gold_labels = life
                else:
                    inst.gold_labels = int(imprisonment)

                inst.words_size = len(inst.words)
                inst.accu_labels_size = len(inst.accu_labels)
                inst.law_labels_size = len(inst.law_labels)
                inst.prison_labels_size = len(inst.prison_labels)
                insts.append(inst)
                if len(insts) == self.max_count:
                    break
            sys.stdout.write("\rreading the {} line\t".format(now_lines))
        return insts





