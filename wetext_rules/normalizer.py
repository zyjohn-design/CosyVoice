# Derived from WeTextProcessing/tn/chinese/normalizer.py.
# Copyright (c) 2022 Zhendong Peng. Licensed under the Apache License, Version 2.0.
"""固定上游语法并在同一个规则清单中注册本地业务规则。"""
from pathlib import Path

from pynini.lib.pynutil import delete
from tn.chinese.normalizer import Normalizer
from tn.chinese.rules.cardinal import Cardinal
from tn.chinese.rules.char import Char
from tn.chinese.rules.date import Date
from tn.chinese.rules.fraction import Fraction
from tn.chinese.rules.math import Math
from tn.chinese.rules.measure import Measure
from tn.chinese.rules.money import Money
from tn.chinese.rules.postprocessor import PostProcessor
from tn.chinese.rules.range import Range
from tn.chinese.rules.sport import Sport
from tn.chinese.rules.time import Time
from tn.chinese.rules.whitelist import Whitelist
from tn.processor import RuleSpec

from rules import Identifier, constrain_identifier_boundaries


class BusinessNormalizer(Normalizer):
    def __init__(self, data_dir: Path, remove_erhua=False):
        self.data_dir = data_dir
        super().__init__(cache_dir=False, remove_erhua=remove_erhua,
                         traditional_to_simple=False, remove_interjections=False,
                         remove_puncts=False, full_to_half=False, tag_oov=False)

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        range_rule = Range()
        identifier = Identifier(self.data_dir)
        rules = (
            RuleSpec(identifier, 1.0),
            RuleSpec(Date(range_tagger=range_rule.tagger), 1.02),
            RuleSpec(Whitelist(remove_erhua=self.remove_erhua), 1.03),
            RuleSpec(Sport(cardinal=cardinal), 1.04),
            RuleSpec(Fraction(cardinal=cardinal), 1.05),
            RuleSpec(Measure(cardinal=cardinal), 1.05),
            RuleSpec(Money(cardinal=cardinal), 1.05),
            RuleSpec(Time(range_tagger=range_rule.tagger), 1.05),
            RuleSpec(cardinal, 1.06),
            RuleSpec(Math(cardinal=cardinal), 90),
            RuleSpec(Char(), 100),
            RuleSpec(range_rule),
        )
        self.tagger = self.tagger_union(rules).star @ self.build_rule(delete(' '), r='[EOS]')
        self.tagger = constrain_identifier_boundaries(self.tagger, identifier)
        processor = PostProcessor(remove_interjections=self.remove_interjections,
                                  traditional_to_simple=self.traditional_to_simple,
                                  remove_puncts=self.remove_puncts,
                                  full_to_half=self.full_to_half, tag_oov=self.tag_oov).processor
        self.verbalizer = (self.verbalizer_union(rules) @ processor).star
