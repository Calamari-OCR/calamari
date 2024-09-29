import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type, Optional, Dict, Tuple, Union, Callable

from dataclasses_json import dataclass_json
from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams
from tfaip.util.enum import StrEnum

from calamari_ocr.utils import resource_filename
from calamari_ocr.ocr.dataset.textprocessors import TextProcessor

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class Replacement:
    # TODO(christoph): remove replacement and join it with Rule
    old: str = ""
    new: str = ""
    regex: bool = False

    def make_fn(self) -> Callable[[str], str]:
        if self.regex:
            regex = re.compile(self.old)
            return lambda s: regex.sub(self.new, s)
        else:
            return lambda s: s.replace(self.old, self.new)


@pai_dataclass
@dataclass
class TextRegularizerProcessorParams(DataProcessorParams):
    replacements: Optional[List[Replacement]] = field(default=None, metadata=pai_meta(mode="ignore"))
    rulesets: List[str] = field(default_factory=lambda: ["spaces"])
    rulegroups: List[str] = field(default_factory=list)

    @staticmethod
    def cls() -> Type["TextProcessor"]:
        return TextRegularizerProcessor


class TextRegularizerProcessor(TextProcessor[TextRegularizerProcessorParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.params.replacements is None:
            self.params.replacements = self._load_rule_groups() + self._load_rule_sets()

        self._functions = None

    @property
    def functions(self):
        if self._functions is None:
            self._functions = [r.make_fn() for r in self.params.replacements]
        return self._functions

    def _load_rule_groups(self) -> List[Replacement]:
        return [r.to_replacement() for r in sum([parse_groupset(r) for r in self.params.rulegroups], [])]

    def _load_rule_sets(self) -> List[Replacement]:
        return [r.to_replacement() for r in sum([parse_ruleset(r) for r in self.params.rulesets], [])]

    def _apply_single(self, txt, meta):
        for replacement in self.functions:
            txt = replacement(txt)

        return txt


class RuleTypes(StrEnum):
    RAW = "raw"
    REGEX = "regex"


@dataclass_json
@dataclass
class Rule:
    rule: Tuple[str, str]  # From -> to
    type: RuleTypes

    def to_replacement(self) -> Replacement:
        if self.type == RuleTypes.RAW:
            return Replacement(*self.rule, regex=False)
        elif self.type == RuleTypes.REGEX:
            return Replacement(*self.rule, regex=True)
        else:
            raise NotImplementedError(f"Unknown RuleType {self.type}")


def parse_groupset(r: str) -> List[Rule]:
    if r not in default_rule_groups:
        raise KeyError(f"Rule group {r} is unknown. Available groups: {list(default_rule_groups.keys())}.")

    return sum(map(parse_ruleset, default_rule_groups[r]), [])


def parse_ruleset(r: Union[List[dict], Path, str]) -> List[Rule]:
    if isinstance(r, str):
        if r.endswith(".json"):
            r = Path(r)
        else:
            if r == "*":
                # Support for * to mean all
                return sum([parse_ruleset(ruleset) for ruleset in default_rulesets.keys()], [])
            if r not in default_rulesets:
                raise KeyError(
                    f"Ruleset {r} is not in the defaults {list(default_rulesets.keys())}. "
                    f"Alternatively, specify a path to a `ruleset.json`."
                )
            r = default_rulesets[r]
    if isinstance(r, Path):
        with open(r.as_posix()) as f:
            r = json.load(f)

    return [Rule.from_dict(d) for d in r]


# LOAD DEFAULT RULE SETS AND GROUPS
# =================================

rulesets_dir = resource_filename("calamari_ocr", "resources") / "rulesets"
default_rulesets = {p.stem: p for p in rulesets_dir.iterdir() if p.suffix == ".json"}

rule_groups_file = resource_filename("calamari_ocr", "resources") / "rulegroups.json"
with open(rule_groups_file) as f:
    default_rule_groups: Dict[str, List[str]] = json.load(f)


logger.debug(f"Found default rulesets: {list(default_rulesets.keys())}")
logger.debug(f"Found default rulegroups: {list(default_rule_groups.keys())}")
