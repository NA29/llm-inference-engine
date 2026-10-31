import itertools
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _ids = itertools.count()

    def __init__(self, prompt_token_ids: list[int], params: SamplingParams, arrival_time: float):
        self.seq_id = next(Sequence._ids)
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.params = params
        self.status = SeqStatus.WAITING
        self.block_table: list[int] = []  # physical KV block of each logical block
        self.num_preemptions = 0
        self.arrival_time = arrival_time
        self.first_token_time: float | None = None
        self.finish_time: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]
