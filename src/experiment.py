# src/experiment.py
import random
from env import GridWorld

# ---------- Run one episode ----------
def run_episode(env, policy="greedy", max_steps=100):
    state = env.reset()
    total_reward = 0.0
    steps = 0
    reached_goal = False

    for _ in range(max_steps):
        if policy == "greedy":
            action = env.greedy_action_towards_goal()
        else:
            action = random.choice([0, 1, 2, 3])

        state, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1

        if done:
            reached_goal = True
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "success": reached_goal
    }


# ---------- Run multiple episodes ----------
def run_many(env, episodes=30, policy="greedy"):
    results = []
    for _ in range(episodes):
        results.append(run_episode(env, policy=policy))
    return results


# ---------- Summarize results ----------
def summarize(results):
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    return success_rate, avg_steps, avg_reward


# ---------- MAIN EXPERIMENT ----------
if __name__ == "__main__":
    random.seed(42)

    settings = [
        (5, 5, 4),
        (7, 7, 8),
        (10, 10, 15),
    ]

    for rows, cols, obs in settings:
        print(f"\nGRID {rows}x{cols} | obstacles={obs}")

        # STATIC: obstacles fixed across episodes
        static_env = GridWorld(
            rows=rows,
            cols=cols,
            start=(0, 0),
            goal=(rows - 1, cols - 1),
            num_obstacles=obs,
            dynamic=False
        )
        static_env._generate_obstacles()
        static_results = run_many(static_env, episodes=30, policy="greedy")

        # DYNAMIC: obstacles regenerate every reset
        dynamic_env = GridWorld(
            rows=rows,
            cols=cols,
            start=(0, 0),
            goal=(rows - 1, cols - 1),
            num_obstacles=obs,
            dynamic=True
        )
        dynamic_results = run_many(dynamic_env, episodes=30, policy="greedy")

        print("  STATIC :", summarize(static_results))
        print("  DYNAMIC:", summarize(dynamic_results))

