"""AdPrompter: an RL loop that learns which persuasion strategy to prompt an image model
with, scored by automatic rating and a demographic-bias audit."""
from .agent import PPOAgent, ThompsonAgent
from .bias import CLIPAuditor, MockAuditor, skew
from .env import AdEnv
from .imaging import MockGenerator, SDXLGenerator
from .pipeline import Record, run_campaign
from .prompting import LLMPromptExpander, TemplatePromptBuilder
from .rating import CLIPRater, HumanRater, SimulatedRater
from .strategies import CUES, STRATEGIES
from .types import Product, Variant

__all__ = [
    "AdEnv", "Product", "Variant", "Record", "run_campaign", "STRATEGIES", "CUES",
    "TemplatePromptBuilder", "LLMPromptExpander", "MockGenerator", "SDXLGenerator",
    "SimulatedRater", "HumanRater", "CLIPRater", "MockAuditor", "CLIPAuditor", "skew",
    "ThompsonAgent", "PPOAgent",
]
