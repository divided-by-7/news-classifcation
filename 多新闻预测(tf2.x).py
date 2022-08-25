# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter import *
from tkinter.filedialog import askopenfilename
import os
import collections
import re
import unicodedata
import six
import tensorflow as tf
import pandas as pd
import time

max_seq_len = 256
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ----------------------------------------------------------------------------
# tokenization中的模块
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):

        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class PredictDetect(object):
    def __init__(self):
        self.label_list = ['财经', '房产', '教育', '科技', '军事', '汽车', '体育', '游戏', '娱乐']

        self.model_path = 'config/classification_model.pb'
        self.detection_graph = self._load_model()

        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        self.input_ids = self.detection_graph.get_tensor_by_name('input_ids:0')
        self.input_mask = self.detection_graph.get_tensor_by_name('input_mask:0')
        self.pred_prob = self.detection_graph.get_tensor_by_name('pred_prob:0')

        self.input_map = {"input_ids": self.input_ids, "input_mask": self.input_mask}

        self.tokenizer = FullTokenizer(
            vocab_file='config/vocab.txt',
            do_lower_case=True)

    def _load_model(self):
        with tf.io.gfile.GFile(self.model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        for op in graph.get_operations():
            print(op.name)

        return graph

    def _data_init(self, sentence):
        _exam = self.one_example(sentence)
        _feature = self.convert_single_example(0, _exam, self.label_list, max_seq_len, self.tokenizer)

        return _feature

    def detect(self, sentence):
        features = self._data_init(sentence)

        preds_evaluated = self.sess.run([self.pred_prob], feed_dict={self.input_ids: [features.input_ids],
                                                                     self.input_mask: [features.input_mask]})
        pred = preds_evaluated[0]
        pred_index = pred.argmax(axis=1)[0]
        pred_score = pred[0][pred_index]
        print(pred_index)

        return self.label_list[pred_index], pred_score

    @staticmethod
    def one_example(sentence):
        guid, label = 'pred-0', '#'
        text_a, text_b = sentence, None
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_single_example(self, ex_index, example, label_list_, max_seq_length, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        label_map = {}
        for (i, label) in enumerate(label_list_):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = None
        # if ex_index < 5:
        #     tf.logging.info("*** Example ***")
        #     tf.logging.info("guid: %s" % example.guid)
        #     tf.logging.info("tokens: %s" % " ".join(
        #         [printable_text(x) for x in tokens]))
        #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        feature_ = InputFeatures(
            input_ids=input_ids, input_mask=input_mask,
            segment_ids=segment_ids, label_id=label_id,
            is_real_example=True)
        return feature_


predict_detect = PredictDetect()

pre_data = ''
index1, score1 = predict_detect.detect(pre_data)
# ------------------------------------------------------------------------------------------------------------

root = Tk()
root.minsize(800, 100)  # 最小尺寸
root.maxsize(800, 100)
inputs1 = 0
inputs2 = 0


def func():
    s1 = askopenfilename()
    s1 = str(s1)
    start = time.time()
    data = pd.read_excel(s1)
    index_list = []
    score_list = []

    for index, test in enumerate(data.values):
        root.update()
        print('正在进行第%s条预测' % index)
        e3.set('正在进行第%s条预测' % index)
        inputt = (str(test[2]) + str(test[3]))
        ind, sco = predict_detect.detect(inputt)
        if sco<0.9: #设置独热码阈值为0.5
          ind = '其它'
        index_list.append(ind)
        score_list.append(sco)
    # print(index_list) # 检查输出label

    # print('正在写入excel')
    e3.set('正在写入excel')
    del data['channelName']
    data.insert(1, 'channelName', index_list)
    data.insert(4, 'Probability', score_list)
    data.to_excel('result.xlsx')    # print('运行完成')
    # print('运行时长：', (time.time() - start))

    e3.set('文件已另存为本文件夹下result.xlsx，本次运行时长%f秒' % (time.time() - start) )


# f1 = Frame(root)  # 定义框架
# Label(f1, text=" 请输入Excel.xlsx文件名：").pack(side=LEFT, padx=5, pady=10)
# e1 = StringVar()  # 定义输入框内容
# inputs = Entry(f1, width=20, textvariable=e1)
# inputs.pack(side=LEFT)  # 输入框
# # e1.set('在此处键入需要计算的金额')
# f1.pack()

f3 = Frame(root)  # 定义框架
Label(f3, text="当前状态：").pack(side=LEFT, padx=5, pady=10)
e3 = StringVar()  # 定义输入框内容
entryOut = Entry(f3, width=60, textvariable=e3, state=DISABLED)  # 禁用输入框
entryOut.pack(side=LEFT)
f3.pack()

f4 = Frame(root)  # 定义框架
btnOk = Button(f4, text="单击此处选择文件进行分类", command=func)
btnOk.pack(side=LEFT, padx=15, pady=10)
f4.pack()
root.title("单条新闻分类器")
# 改变窗口图标
# root.iconbitmap('encrypt.ico')
root.mainloop()
