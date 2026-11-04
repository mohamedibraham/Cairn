from enums import ArtifactType
from services.artifacts.base import ArtifactStrategy
from services.artifacts.chart import ChartStrategy
from services.artifacts.diagram import DiagramStrategy
from services.artifacts.flashcards import FlashcardsStrategy
from services.artifacts.keywords import KeywordsStrategy
from services.artifacts.mcq import McqStrategy
from services.artifacts.quiz import QuizStrategy
from services.artifacts.summary import SummaryStrategy

_registry: dict = {}


def register_strategy(strategy: ArtifactStrategy, *, replace: bool = False) -> None:
    if strategy.type in _registry and not replace:
        raise ValueError(f"a strategy for {strategy.type.value!r} is already registered")
    _registry[strategy.type] = strategy


def get_strategy(artifact_type: ArtifactType) -> ArtifactStrategy:
    return _registry[artifact_type]


for _strategy in (KeywordsStrategy(), FlashcardsStrategy(), McqStrategy(), QuizStrategy(),
                  SummaryStrategy(), ChartStrategy(), DiagramStrategy()):
    register_strategy(_strategy)

assert set(_registry) == set(ArtifactType), "every ArtifactType needs a strategy"
