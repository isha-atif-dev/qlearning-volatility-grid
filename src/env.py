# env.py  ----- A mathematical simulation of a maze(environment)


from typing import Tuple, Dict, Set


class GridWorld:
    """
    A simple grid environment for navigation (simulation on laptop).

    Symbols:
      A = agent
      G = goal
      # = obstacle
      . = empty
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]] | None = None
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles is not None else set()

        self.agent_pos = start  # current position

    # ---------- Helper checks ----------
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        return pos in self.obstacles

    # ---------- Core environment functions ----------
    def reset(self) -> Tuple[int, int]:
        """Reset agent to start and return initial state."""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take an action and move the agent if possible.

        Returns:
          next_state: (row, col)
          reward: float
          done: True if reached goal
          info: extra details (optional)
        """
        if action not in self.ACTIONS:
            raise ValueError("Invalid action. Use 0=up,1=down,2=left,3=right.")

        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        proposed = (r + dr, c + dc)

        # Rule 1: if outside grid -> stay, penalty
        if not self._in_bounds(proposed):
            reward = -5.0
            done = False
            info = {"event": "hit_wall"}
            return self.agent_pos, reward, done, info

        # Rule 2: if obstacle -> stay, penalty
        if self._is_blocked(proposed):
            reward = -10.0
            done = False
            info = {"event": "hit_obstacle"}
            return self.agent_pos, reward, done, info

        # Move is valid -> update position
        self.agent_pos = proposed

        # Rule 3: reached goal -> big reward, done
        if self.agent_pos == self.goal:
            reward = 100.0
            done = True
            info = {"event": "reached_goal"}
            return self.agent_pos, reward, done, info

        # Rule 4: normal move -> small step penalty
        reward = -1.0
        done = False
        info = {"event": "moved"}
        return self.agent_pos, reward, done, info

    def render(self) -> None:
        """Print the grid."""
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == self.agent_pos:
                    row_chars.append("A")
                elif cell == self.goal:
                    row_chars.append("G")
                elif cell in self.obstacles:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            print(" ".join(row_chars))
        print()


if __name__ == "__main__":
    # Quick manual test (no RL yet)
    obstacles = {(1, 1), (1, 2), (2, 2)}
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4), obstacles=obstacles)

    state = env.reset()
    env.render()

    # Try some moves:
    # 3=right, 1=down
    actions = [3, 3, 1, 1, 1, 3, 3, 1]  # just a sample
    for a in actions:
        next_state, reward, done, info = env.step(a)
        print(f"Action={a} -> State={next_state}, Reward={reward}, Done={done}, Info={info}")
        env.render()
        if done:
            break
