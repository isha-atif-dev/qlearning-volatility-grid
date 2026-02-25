import matplotlib.pyplot as plt

def plot_learning(rewards, steps, title="Learning curves"):
    # Rewards curve
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title + " - Reward")
    plt.show()

    # Steps curve
    plt.figure()
    plt.plot(steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps to finish (or max)")
    plt.title(title + " - Steps")
    plt.show()
