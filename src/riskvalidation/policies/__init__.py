"""Packaged validation policies (JSON) and the rules their entries name.

Thresholds are policy choices, documented per test, never regulation. The package carries the policy files
(``default.json`` and one file per Contraste battery, loaded by ``riskvalidation.core.policy``) and the module
:mod:`riskvalidation.policies.rules`, whose functions are the rules a policy entry names in its ``rule``. The
module and the rule names are public: a packaged file names them, so removing or renaming one is a major change.
"""

from riskvalidation.policies import rules
from riskvalidation.policies.rules import *  # noqa: F403  (the rule names, each in rules.__all__)

__all__ = ["rules", *rules.__all__]
