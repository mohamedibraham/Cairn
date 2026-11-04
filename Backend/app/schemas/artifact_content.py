from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

ChartType = Literal["bar", "line", "pie", "scatter"]


SERIES_COLORS = ("var(--primary)", "var(--trace)", "var(--success)", "var(--warning)")


class _Content(BaseModel):
    
    model_config = ConfigDict(extra="allow")


class SourceRef(BaseModel):
    chunk_id: str
    file_id: str
    filename: str
    page_number: int | None = None
    line_start: int | None = None
    symbol: str | None = None


# summary ---------------------------------------------------------------------

class SummaryContent(_Content):
    type: Literal["summary"] = "summary"
    body: str = Field(min_length=1)  
    sources: list[SourceRef] = []


# keywords --------------------------------------------------------------------

class KeywordItem(_Content):
    term: str = Field(min_length=1)
    weight: float | None = Field(default=None, ge=0, le=1)
    definition: str | None = None
    source: str | None = None  

class KeywordsContent(_Content):
    type: Literal["keywords"] = "keywords"
    items: list[KeywordItem] = Field(min_length=1)
    sources: list[SourceRef] = []


#  flashcards ------------------------------------------------------------------

class FlashcardItem(_Content):
    id: str
    front: str = Field(min_length=1)
    back: str = Field(min_length=1)
    source: str | None = None


class FlashcardsContent(_Content):
    type: Literal["flashcards"] = "flashcards"
    cards: list[FlashcardItem] = Field(min_length=1)
    sources: list[SourceRef] = []


# mcq ---------------------------------------------------------------------------

class McqQuestion(_Content):
    id: str
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    correct_index: int = Field(ge=0)
    explanation: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index is out of range")
        if len({o.strip().casefold() for o in self.options}) != len(self.options):
            raise ValueError("options must be distinct")
        return self


class McqContent(_Content):
    type: Literal["mcq"] = "mcq"
    questions: list[McqQuestion] = Field(min_length=1)
    sources: list[SourceRef] = []


# quiz --------------------------------------------------------------------------

QuizKind = Literal["open", "true_false", "mcq", "fill_blank"]
BLANK = "____"


class QuizQuestion(_Content):
    
    id: str
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    kind: QuizKind = "open"
    options: list[str] | None = None
    correct_index: int | None = None
    accepted_answers: list[str] | None = None
    explanation: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "true_false" and self.answer not in ("True", "False"):
            raise ValueError('true_false answer must be "True" or "False"')
        if self.kind == "mcq":
            if not self.options or len(self.options) < 2 or self.correct_index is None:
                raise ValueError("mcq needs options and correct_index")
            if not 0 <= self.correct_index < len(self.options):
                raise ValueError("correct_index is out of range")
        if self.kind == "fill_blank":
            if self.question.count(BLANK) != 1:
                raise ValueError(f"fill_blank needs exactly one {BLANK}")
            if not self.accepted_answers:
                raise ValueError("fill_blank needs accepted_answers")
        return self


class QuizContent(_Content):
    type: Literal["quiz"] = "quiz"
    questions: list[QuizQuestion] = Field(min_length=1)
    sources: list[SourceRef] = []


# chart -------------------------------------------------------------------------

class ChartSeries(BaseModel):
    key: str
    label: str
    color: str


class ChartContent(_Content):
    
    type: Literal["chart"] = "chart"
    chart_type: ChartType
    x_key: str
    x_label: str | None = None
    y_label: str | None = None
    caption: str | None = None
    data: list[dict[str, Union[str, float, int]]] = Field(min_length=1)
    series: list[ChartSeries] = Field(min_length=1)
    sources: list[SourceRef] = []

    @model_validator(mode="after")
    def _check(self):
        if self.chart_type == "pie" and len(self.series) != 1:
            raise ValueError("a pie chart has exactly one series")
        keys = [s.key for s in self.series]
        if len(set(keys)) != len(keys) or self.x_key in keys:
            raise ValueError("series keys must be unique and different from x_key")
        for row in self.data:
            if self.x_key not in row:
                raise ValueError(f"every row needs {self.x_key!r}")
        for key in keys:
            if not any(key in row for row in self.data):
                raise ValueError(f"series {key!r} has no values")
        return self


# diagram -----------------------------------------------------------------------

class DiagramNode(BaseModel):
    id: str
    label: str = Field(min_length=1)
    x: float
    y: float


class DiagramEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str | None = None


class DiagramContent(_Content):
    
    type: Literal["diagram"] = "diagram"
    nodes: list[DiagramNode] = Field(min_length=1)
    edges: list[DiagramEdge] = []
    sources: list[SourceRef] = []

    @model_validator(mode="after")
    def _check(self):
        node_ids = [n.id for n in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node ids must be unique")
        edge_ids = [e.id for e in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("edge ids must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge {edge.id!r} points at a missing node")
        return self


ArtifactContent = Annotated[
    Union[SummaryContent, KeywordsContent, FlashcardsContent, McqContent, QuizContent, ChartContent, DiagramContent],
    Field(discriminator="type"),
]

_adapter = TypeAdapter(ArtifactContent)


def parse_content(data: dict):
    return _adapter.validate_python(data)
