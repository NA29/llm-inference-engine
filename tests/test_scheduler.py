from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.scheduler import Scheduler
from engine.sequence import SamplingParams, Sequence, SeqStatus


def make(cfg: EngineConfig, num_blocks: int, block_size: int = 4):
    bm = BlockManager(num_blocks, block_size, cfg.kv_allocation, cfg.max_model_len)
    return Scheduler(cfg, bm, eos_token_ids=[999]), bm


def seq(n_prompt: int, max_tokens: int = 8) -> Sequence:
    return Sequence(list(range(n_prompt)), SamplingParams(max_tokens=max_tokens), arrival_time=0.0)


def run_step(sched: Scheduler, token: int = 1):
    seqs, is_prefill = sched.schedule()
    sched.postprocess(seqs, [token] * len(seqs))
    return seqs, is_prefill


def test_new_requests_join_running_batch():
    sched, _ = make(EngineConfig(max_num_seqs=8), num_blocks=100)
    a = seq(4)
    sched.add(a)
    assert run_step(sched) == ([a], True)
    assert run_step(sched) == ([a], False)
    b = seq(4)
    sched.add(b)
    assert run_step(sched) == ([b], True)  # b is admitted while a is mid-generation
    seqs, is_prefill = run_step(sched)
    assert set(seqs) == {a, b} and not is_prefill


def test_finished_sequences_leave_and_free_blocks():
    sched, bm = make(EngineConfig(), num_blocks=100)
    a = seq(4, max_tokens=2)
    sched.add(a)
    run_step(sched)
    run_step(sched)
    assert a.status == SeqStatus.FINISHED and not sched.has_unfinished()
    assert bm.num_free_blocks == 99


def test_eos_stops_sequence():
    sched, _ = make(EngineConfig(), num_blocks=100)
    a = seq(4, max_tokens=50)
    sched.add(a)
    run_step(sched)
    run_step(sched, token=999)
    assert a.status == SeqStatus.FINISHED and a.output_token_ids == [1, 999]


def test_admission_waits_for_free_blocks():
    sched, _ = make(EngineConfig(), num_blocks=3)  # 2 usable blocks of 4 tokens
    a, b = seq(8), seq(8)
    sched.add(a)
    sched.add(b)
    seqs, _ = sched.schedule()
    assert seqs == [a] and list(sched.waiting) == [b]


def test_preempts_newest_when_out_of_blocks():
    sched, bm = make(EngineConfig(), num_blocks=3)  # 2 usable blocks
    a, b = seq(4), seq(4)
    sched.add(a)
    sched.add(b)
    run_step(sched)  # both prefilled, one block each, pool empty
    # both now have 5 tokens and need a second block; only one can get it
    seqs, _ = run_step(sched)
    assert seqs == [a]
    assert b.status == SeqStatus.WAITING and b.num_preemptions == 1 and b.block_table == []
    assert sched.waiting[0] is b


def test_static_batching_waits_for_whole_batch():
    sched, _ = make(EngineConfig(scheduling="static"), num_blocks=100)
    a = seq(4, max_tokens=3)
    sched.add(a)
    run_step(sched)            # prefill a
    b = seq(4)
    sched.add(b)
    assert run_step(sched) == ([a], False)  # b must wait for a's batch to finish
    run_step(sched)            # a finishes
    assert run_step(sched) == ([b], True)
