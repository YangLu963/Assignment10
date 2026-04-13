"""Part 5 (stretch): Paged KV Cache

Fixed-size blocks instead of contiguous allocation.
Same idea as OS virtual memory pages.

Key idea:
- Memory is divided into fixed-size blocks (e.g. 16 tokens per block)
- Each request gets allocated only as many blocks as it needs
- When a request finishes, its blocks are returned to the free pool
- This avoids wasting memory from pre-allocating max_seq_len for every request
"""

import math
import torch


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_heads: int, head_dim: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers

        # Physical memory pools for keys and values
        # Shape: [num_blocks, num_heads, block_size, head_dim] per layer
        self.k_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        # All blocks start as free
        self.free_blocks: list[int] = list(range(num_blocks))

        # Maps seq_id -> list of block IDs allocated to that sequence
        self.seq_to_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence. Returns list of block IDs.

        We need ceil(num_tokens / block_size) blocks to store num_tokens tokens.
        Blocks are taken from the free pool and recorded under seq_id.
        Raises RuntimeError if there are not enough free blocks.
        """
        # Calculate how many blocks we need
        num_blocks_needed = math.ceil(num_tokens / self.block_size)

        # Check we have enough free blocks
        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free blocks: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        # Take blocks from the free pool
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)

        # Record which blocks belong to this sequence
        self.seq_to_blocks[seq_id] = allocated

        return allocated

    def free(self, seq_id: int):
        """Free all blocks for a finished sequence.

        Returns the blocks back to the free pool so other sequences can use them.
        Does nothing if seq_id was never allocated.
        """
        if seq_id not in self.seq_to_blocks:
            return

        # Get the blocks that belong to this sequence
        blocks_to_free = self.seq_to_blocks.pop(seq_id)

        # Zero out the memory in those blocks (clean up for next user)
        for block_id in blocks_to_free:
            for layer_idx in range(self.num_layers):
                self.k_pool[layer_idx][block_id].zero_()
                self.v_pool[layer_idx][block_id].zero_()

        # Return blocks to the free pool
        self.free_blocks.extend(blocks_to_free)

    def get_block_ids(self, seq_id: int) -> list[int]:
        """Return the block IDs allocated to a sequence."""
        return self.seq_to_blocks.get(seq_id, [])

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if there are enough free blocks for num_tokens tokens."""
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        return num_blocks_needed <= len(self.free_blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)