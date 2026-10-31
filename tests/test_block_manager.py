from engine.block_manager import BlockManager
from engine.sequence import SamplingParams, Sequence


def make_seq(n_prompt: int) -> Sequence:
    return Sequence(list(range(n_prompt)), SamplingParams(max_tokens=100), arrival_time=0.0)


def test_allocates_on_demand_and_frees():
    bm = BlockManager(num_blocks=11, block_size=16)  # block 0 is reserved scratch
    assert bm.num_free_blocks == 10
    seq = make_seq(20)  # 20 tokens -> 2 blocks
    bm.allocate(seq)
    assert len(seq.block_table) == 2 and 0 not in seq.block_table
    # decode tokens 21..32 fit in block 2; token 33 needs a third block
    for _ in range(12):
        seq.output_token_ids.append(1)
        bm.append_slot(seq)
    assert len(seq.block_table) == 2
    seq.output_token_ids.append(1)
    bm.append_slot(seq)
    assert len(seq.block_table) == 3
    bm.free(seq)
    assert bm.num_free_blocks == 10 and seq.block_table == []


def test_can_append_when_full():
    bm = BlockManager(num_blocks=3, block_size=4)
    seq = make_seq(8)
    bm.allocate(seq)
    assert bm.num_free_blocks == 0
    seq.output_token_ids.append(1)  # token 9 needs a new block
    assert not bm.can_append(seq)


def test_prealloc_reserves_max_model_len():
    bm = BlockManager(num_blocks=1 + 2 * 128, block_size=16, mode="prealloc", max_model_len=2048)
    a, b, c = make_seq(10), make_seq(10), make_seq(10)
    bm.allocate(a)
    assert len(a.block_table) == 128  # 2048 / 16, regardless of the 10-token prompt
    assert bm.can_allocate(b)
    bm.allocate(b)
    assert not bm.can_allocate(c)


def test_prealloc_oracle_reserves_prompt_plus_max_tokens():
    bm = BlockManager(num_blocks=100, block_size=16, mode="prealloc_oracle")
    seq = make_seq(20)  # 20 + 100 max_tokens = 120 tokens -> 8 blocks
    bm.allocate(seq)
    assert len(seq.block_table) == 8
