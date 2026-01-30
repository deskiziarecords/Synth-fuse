# src/synthfuse/forum/protocols/turn.py
from typing import List, Iterator
from synthfuse.agents.base import Agent

class TurnManager:
    """Manages speaking order in debate."""
    def __init__(self, agents: List[Agent], max_rounds: int = 3):
        self.agents = agents
        self.max_rounds = max_rounds
        self.history = []

    def __iter__(self) -> Iterator[tuple[Agent, int]]:
        for round_num in range(self.max_rounds):
            for agent in self.agents:
                yield agent, round_num
