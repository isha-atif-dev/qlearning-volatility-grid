# src/experiment.py
import random
from env import GridWorld
from agent import QLearningAgent
import os
import time
from metrics import plot_learning

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def play_episode(env, choose_action_fn, max_steps=100, delay=0.25, title=""):
    """
    Visualise one episode by rendering the grid each step.
    choose_action_fn(state) -> action
    """
    state = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        clear_screen()
        if title:
            print(title)
        print(f"step={step}  state={state}  total_reward={total_reward:.1f}")
        env.render()
        time.sleep(delay)

        action = choose_action_fn(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

        if done:
            clear_screen()
            if title:
                print(title)
            print(f"✅ reached goal in {step+1} steps | total_reward={total_reward:.1f}")
            env.render()
            time.sleep(1.0)
            break

    return total_reward

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




def train_q_learning(env, episodes=500, max_steps=100, alpha=0.1, gamma=0.9, epsilon=0.2):
    agent = QLearningAgent(actions=env.ACTIONS.keys(), alpha=alpha, gamma=gamma, epsilon=epsilon)

    rewards_per_episode = []
    steps_per_episode = []
    success_count = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                success_count += 1
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step + 1)

    return agent, rewards_per_episode, steps_per_episode, success_count / episodes





def evaluate_agent(env, agent, episodes=30, max_steps=100):
    # turn off exploration for evaluation
    agent.epsilon = 0.0

    results = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        success = False

        for _ in range(max_steps):
            action = agent.best_action(state)   # IMPORTANT: use learned best move
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                success = True
                break

        results.append({"steps": steps, "total_reward": total_reward, "success": success})

    return summarize(results)

# ---------- MAIN EXPERIMENT ----------
if __name__ == "__main__":
    random.seed(42)

    rows, cols, obs = 5, 5, 4
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    # ----------------------------
    # A) Train on STATIC
    # ----------------------------
    static_train_env = GridWorld(rows, cols, start, goal, num_obstacles=obs, dynamic=False)
    static_train_env._generate_obstacles()

    static_agent, rewards, steps, static_train_success = train_q_learning(static_train_env, episodes=500)
    print("STATIC training success rate:", static_train_success)

    plot_learning(rewards, steps, title="Q-learning training (static)")
    static_eval_env = GridWorld(rows, cols, start, goal, num_obstacles=obs, dynamic=False)
    static_eval_env.obstacles = static_train_env.obstacles.copy()

    dynamic_eval_env = GridWorld(rows, cols, start, goal, num_obstacles=obs, dynamic=True)

    print("STATIC-trained agent | STATIC eval :", evaluate_agent(static_eval_env, static_agent, episodes=30))
    print("STATIC-trained agent | DYNAMIC eval:", evaluate_agent(dynamic_eval_env, static_agent, episodes=30))

    # ----------------------------
    # B) Train on DYNAMIC
    # ----------------------------
    dynamic_train_env = GridWorld(rows, cols, start, goal, num_obstacles=obs, dynamic=True)

    dynamic_agent, rewards, steps, dynamic_train_success = train_q_learning(dynamic_train_env, episodes=1500)
    print("\nDYNAMIC training success rate:", dynamic_train_success)

    # evaluate dynamic-trained agent
    static_eval_env2 = GridWorld(rows, cols, start, goal, num_obstacles=obs, dynamic=False)
    static_eval_env2._generate_obstacles()  # any fixed static map

    dynamic_eval_env2 = GridWorld(rows, cols, start, goal, num_obstacles=obs, dynamic=True)

    print("DYNAMIC-trained agent | STATIC eval :", evaluate_agent(static_eval_env2, dynamic_agent, episodes=30))
    print("DYNAMIC-trained agent | DYNAMIC eval:", evaluate_agent(dynamic_eval_env2, dynamic_agent, episodes=30))

    # Visualise GREEDY in a dynamic environment (one episode)
    play_episode(
        dynamic_eval_env,
        choose_action_fn=lambda s: dynamic_eval_env.greedy_action_towards_goal(),
        max_steps=100,
        delay=0.25,
        title="GREEDY | Dynamic environment"
    )
    # Visualise Q-LEARNING (trained) in static environment
    static_agent.epsilon = 0.0
    play_episode(
        static_eval_env,
        choose_action_fn=lambda s: static_agent.best_action(s),
        max_steps=100,
        delay=0.25,
        title="Q-LEARNING | Static environment (trained)"
    )

    # Visualise Q-LEARNING (trained on static) in dynamic environment
    static_agent.epsilon = 0.0
    play_episode(
        dynamic_eval_env,
        choose_action_fn=lambda s: static_agent.best_action(s),
        max_steps=100,
        delay=0.25,
        title="Q-LEARNING | Dynamic environment (trained on static)"
    )



