from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEnv(ABC):
    """Unified controller-agent interaction environment interface.

    Both replay simulator env and real search-o1 env should implement this API.
    """

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        """Start a new episode and return (initial_state, info).

        state shape: [K=5, signal_dim=5]
        """

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """Execute controller action and return (next_state, reward, done, info).

        Action codes: 0=Continue, 1=Compress, 2=Redirect, 3=ModeSwitch, 4=Stop
        Reward: intermediate steps are 0; terminal step returns trajectory-level reward.
        """

    @abstractmethod
    def close(self) -> None:
        """Close environment resources."""
