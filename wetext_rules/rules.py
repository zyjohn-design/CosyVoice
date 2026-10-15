"""在 WeTextProcessing 中注册的原生 WFST 规则，仅供编译使用。"""
from pathlib import Path

import pynini
from pynini.lib import byte, utf8
from pynini.lib.pynutil import delete
from tn.processor import Processor


class Identifier(Processor):
    """字段名 + 可选连接词/冒号 + 数字串，保留原始字段值供 verbalizer 转换。"""

    def __init__(self, data_dir: Path):
        super().__init__('identifier')
        fields = pynini.string_file(str(data_dir/'identifier_fields.tsv'))
        digits = pynini.string_file(str(data_dir/'digits.tsv'))
        spaces = delete(pynini.union(' ', '\t', '　')).star
        connector = pynini.union('为', '是').ques
        colon = pynini.union(':', '：').ques
        self.graph = (fields + spaces + connector + spaces + colon + spaces + digits.plus).optimize()
        self.tagger = self.add_tokens(self.tag_field('value', self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field('value', self.graph))


def constrain_identifier_boundaries(tagger, identifier):
    """在分类图中排除只识别混合编号前半段的路径，不改动输入文本。

    例如 1024ABC、2026-001、1024.5 应由原有规则完整处理，不能将
    前几个数字匹配成 identifier，再将剩余字符当成其他类型。
    """
    sigma = utf8.VALID_UTF8_CHAR.star
    token_name = pynini.union(byte.ALPHA, '_').plus
    value_prefix = token_name + ' { ' + token_name + ': "'
    unsupported = pynini.union(byte.ALNUM, '_', '-', '/')
    next_token = value_prefix + unsupported
    # 点号只有后面继续跟编号字符时才表示小数/复合编号，句末点号保留。
    dot_continuation = (value_prefix + '.' + unsupported |
                        'char { value: "." } ' + value_prefix + unsupported)
    comma_continuation = (value_prefix + pynini.union(',', '，') + byte.DIGIT |
                          pynini.union('char { value: "," } ', 'char { value: "，" } ') +
                          value_prefix + byte.DIGIT)
    identifier_output = identifier.tagger.copy().project('output')
    forbidden = sigma + identifier_output + pynini.accep(' ').star + (next_token | dot_continuation | comma_continuation) + sigma
    return (tagger @ pynini.difference(sigma, forbidden)).optimize()
