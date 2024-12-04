from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llm_genjax.envs import simulate



def eval_simulation(states, fps, save_dir=None):
    """
    Args:
        states: pytree of states
        fps: frames per second
    """
    food_collected = 0
    time_in_nest = 0
    num_steps = len(states.agent_pos)
    for t in range(num_steps):
        if states.eating_food[t]:
            food_collected += 1
        if np.linalg.norm(states.agent_pos[t] - states.nest_pos[t]) < 10.0:
            time_in_nest += 1 / fps

    total_distance = 0
    speed_list = []
    for t in range(num_steps):
        speed = np.linalg.norm(states.agent_pos[t] - states.prev_agent_pos[t])
        total_distance += speed
        speed_list.append(speed)
    # mean_speed = total_distance / num_steps
    mean_speed = np.mean(speed_list)
    max_speed = np.max(speed_list)
    min_speed = np.min(speed_list)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # plot agent position over time
        plt.figure(figsize=(5.5, 2.5), dpi=300)
        plt.plot(states.agent_pos[:, 0])
        plt.plot(states.agent_pos[:, 1])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig(save_dir / "agent_pos.png", bbox_inches='tight')
        plt.close()
    

    return {
        "food_collected": float(food_collected),
        "time_in_nest": float(time_in_nest),
        "max_speed": float(max_speed),
        "min_speed": float(min_speed),
        "mean_speed": float(mean_speed),
    }


def eval_agents(rng, env, gt_agent, agents, num_steps, fps, save_dir, choice_maps=None):
    if (save_dir / "results.json").exists():
        print(f"Loading results from {save_dir / 'results.json'}")
        results = json.load(open(save_dir / "results.json"))

    else:
        results = {}

    save_dir.mkdir(parents=True, exist_ok=True)

    choice_maps = choice_maps or {}

    assert "agent/groundtruth" not in agents
    agents = {**agents, "agent/groundtruth": gt_agent}
    for name, agent in agents.items():
        if name in results:
            # load existing results for that agent
            continue

        _save_dir = save_dir / name.split("/")[-1]
        _, states, actions = simulate(rng, env, agent, num_steps=num_steps, choice_map=choice_maps.get(name, None))
        results[name] = eval_simulation(states, fps=fps, save_dir=_save_dir)

    # save results
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    results_df = pd.DataFrame(results).T
    # add a column for the agent name
    results_df["agent"] = results_df.index
    results_df.to_csv(save_dir / "results.csv")

    gt_results_df = results_df.loc[["agent/groundtruth"]]

    # scatter plot of gt food collected vs time in nest
    plt.figure(figsize=(5.5, 2.5), dpi=300)

    # Plot ground truth agents as larger markers
    # sns.scatterplot(
    #     data=gt_results_df, 
    #     x="food_collected", 
    #     y="time_in_nest",
    #     hue="agent",
    #     marker='o',
    #     s=100,       # larger size
    #     legend='brief'
    # )
    # # Plot best agent runs as smaller dots with same colors
    # sns.scatterplot(
    #     data=results_df.drop(index=["agent/groundtruth"]),
    #     x="food_collected",
    #     y="time_in_nest", 
    #     hue="agent",
    #     marker='$\circ$',  # open circle marker for best agents
    #     s=80,        # smaller size
    #     legend=False # don't add duplicate legend entries
    # )
    # Plot best agent runs as smaller dots with same colors
    sns.scatterplot(
        data=results_df,
        x="food_collected",
        y="time_in_nest", 
        hue="agent",
        s=80,        # smaller size
    )
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Amount of food collected")
    plt.ylabel("Time in nest")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # # Get the current handles and labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    
    # # Add marker-type indicators to the legend
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
    #            label='Groundtruth', markersize=10),
    #     Line2D([0], [0], marker='$\circ$', color='w', markerfacecolor='gray',
    #            label='LLM-generated', markersize=10)
    # ]
    
    # # Combine both types of legend elements
    # all_handles = legend_elements + handles
    # all_labels = ['Groundtruth', 'LLM-generated'] + labels
    
    # # Create new legend
    # ax = plt.gca()
    # ax.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / "eval.png", bbox_inches='tight')
    plt.close()