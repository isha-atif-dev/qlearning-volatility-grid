# env.py  ----- A mathematical simulation of a maze(environment)
from typing import Tuple, Dict, Set
import random
import time
import os

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
        1: (1, 0),  # down
        2: (0, -1),  # left
        3: (0, 1),  # right
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        num_obstacles: int = 3,
        dynamic: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.num_obstacles = num_obstacles
        self.dynamic = dynamic

        self.agent_pos = start
        self.obstacles = set()

    # ---------- Helper checks ----------
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        return pos in self.obstacles

    # ---------- Core environment functions ----------
    def reset(self):
        self.agent_pos = self.start

        if self.dynamic:
            self._generate_obstacles()

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

    def _generate_obstacles(self):
        self.obstacles = set()

        while len(self.obstacles) < self.num_obstacles:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            pos = (r, c)

            # Do NOT place obstacle on start or goal
            if pos != self.start and pos != self.goal:
                self.obstacles.add(pos)



    def greedy_action_towards_goal(self) -> int:
        """
        Simple baseline: move closer to the goal if possible.
        Not learning. Just greedy.
        """
        ar, ac = self.agent_pos
        gr, gc = self.goal

        # try moves in order of preference (reduce distance)
        candidates = []
        if gr < ar: candidates.append(0)  # up
        if gr > ar: candidates.append(1)  # down
        if gc < ac: candidates.append(2)  # left
        if gc > ac: candidates.append(3)  # right

        # if preferred moves blocked, try any move
        if not candidates:
            candidates = [0, 1, 2, 3]

        # pick the first candidate that doesn't hit wall/obstacle
        for a in candidates:
            dr, dc = self.ACTIONS[a]
            nr, nc = ar + dr, ac + dc
            pos = (nr, nc)
            if self._in_bounds(pos) and not self._is_blocked(pos):
                return a

        # if all blocked (rare), just return something
        return random.choice([0, 1, 2, 3])


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    env = GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        num_obstacles=4,
        dynamic=True
    )

    steps_per_episode = 10

    for episode in range(3):
        env.reset()
        print(f"Episode {episode + 1}")
        env.render()

        for _ in range(steps_per_episode):
            clear()
            env.render()
            time.sleep(0.4)

            action = env.greedy_action_towards_goal()
            state, reward, done, info = env.step(action)

            
            if done:
                clear()
                env.render()
                print("Reached Goal!")
                time.sleep(1)
                break


if __name__ == "__main__":
    main()
