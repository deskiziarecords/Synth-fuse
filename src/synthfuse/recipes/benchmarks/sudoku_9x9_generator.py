# src/synthfuse/recipes/benchmarks/sudoku_9x9_generator.py
"""Generate valid 9x9 Sudoku puzzles for ELIXIR 3 benchmarking."""

import random
from typing import List, Tuple


def generate_complete_grid() -> List[List[int]]:
    """Generate a complete, valid Sudoku solution."""
    # Simplified: use pattern-based generation
    base = 3
    side = base * base
    
    # Pattern for Latin square
    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side
    
    # Randomize rows, cols, nums
    from random import sample
    r_base = range(base)
    rows = [g * base + r for g in sample(r_base, len(r_base)) for r in sample(r_base, len(r_base))]
    cols = [g * base + c for g in sample(r_base, len(r_base)) for c in sample(r_base, len(r_base))]
    nums = sample(range(1, side + 1), side)
    
    # Produce board
    board = [[nums[pattern(r, c)] for c in cols] for r in rows]
    return board


def remove_numbers(grid: List[List[int]], difficulty: str = 'medium') -> str:
    """Remove numbers to create puzzle."""
    # Difficulty = number of clues (given cells)
    clues = {'easy': 40, 'medium': 30, 'hard': 25}[difficulty]
    
    puzzle = [row[:] for row in grid]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    
    for r, c in cells[:81 - clues]:
        puzzle[r][c] = 0
    
    # Convert to string
    return ''.join(str(cell) for row in puzzle for cell in row)


def generate_sudoku(difficulty: str = 'medium') -> str:
    """Generate a valid Sudoku puzzle string."""
    solution = generate_complete_grid()
    puzzle = remove_numbers(solution, difficulty)
    return puzzle


if __name__ == "__main__":
    # Generate sample
    for diff in ['easy', 'medium', 'hard']:
        puzzle = generate_sudoku(diff)
        print(f"{diff}: {puzzle[:20]}... (length {len(puzzle)})")
