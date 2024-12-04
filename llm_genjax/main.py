from pathlib import Path

import jax
import jax.tree_util as jtu

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import ChoiceMap
from matplotlib import pyplot as plt

from llm_genjax import make, register
from llm_genjax.envs import animate_frames, make_model, simulate
from llm_genjax.interpreter import Interpreter
from llm_genjax.llm_utils import query
from llm_genjax.eval import eval_agents


save_dir = Path(__file__).parent / "results" / Path(__file__).stem
prompt_dir = Path(__file__).parent / "prompts"

instructions = (prompt_dir / "instructions.txt").read_text()
env_interface = (prompt_dir / "env_interface.txt").read_text()
genjax_docs = (prompt_dir / "genjax_docs.txt").read_text()
agent_code = (prompt_dir / "agent_code.txt").read_text()


@register("description/agent/nearest_food")
def agent_description():
    return "Implement a simple agent that moves towards the nearest food. Sample the speed of the agent in the init function. Use a uniform prior distribution between 0 and 10 for the speed."


@register("description/agent/nearest_food-sleep")
def agent_description_sleep():
    # edit: add info about orders of magnitude (the llm could infer that from statistics)
    # edit: add that when both the food and sleep levels are high, the agent collects food
    # edit: have to specify that the agent has an internal state (foraging or not), it doesn't just react to the its current food and sleep levels
    # return "Implement an agent that moves towards the nearest food. When the sleep level of the agent is low, the agent goes back to its nest. The agent starts collecting food again when it's food level is low. When both the food and sleep levels are high, the agent collects food. Sample all the unknown variables in the agent init function. Use the agent's parameters to specify the prior distributions. Use this information to implement the prior distributions: the speed of the agent is between 0 and 10, the food level is between 0 and 1, and the sleep level is between 0 and 1."

    # aksed claude to describe the gt agent from the code (description above is not accurate enough and not able to infer the init state because the sleep mechanism is not exactly like the gt agent)
    # edit: add that should use the init state for the prior distributions
    # edit: add that the agent starts in foraging mode
    return """
This is a foraging agent that alternates between two behavioral modes: resting and seeking food. The agent uses a PD (Proportional-Derivative) controller for movement, allowing smooth navigation toward target positions.
When in FOOD mode, the agent seeks out the nearest food source. When in REST mode, it returns to its nest. The mode switching is governed by two key factors: hunger (tracked by food_level) and tiredness (tracked by sleep_level). The agent will prioritize resting when tired, but will interrupt rest to seek food if it becomes too hungry. However, if both tired and hungry simultaneously, the agent will prioritize resting over feeding. The agent starts in FOOD mode.
The movement control uses a PD controller that calculates appropriate velocities based on the error (distance to target) and the rate of change of that error. This creates smooth, damped motion toward targets while preventing overshooting. The controller also includes velocity magnitude limiting to maintain a maximum speed, and a threshold-based stopping mechanism when sufficiently close to targets.
Sample the agent speed and the food and sleep level thresholds in the agent init function (variables: 'speed', 'food_threshold', 'sleep_threshold'). Use the agent's parameters to specify the prior distributions. Use this information to implement the prior distributions: the speed of the agent is between 0 and 10, the food level is between 0 and 1, and the sleep level is between 0 and 1.
"""

@register("description/agent/nearest_food-sleep-variant1")
def agent_description_sleep_variant1():
    # remove 'The agent starts in FOOD mode.'
    desc = agent_description_sleep()
    assert " The agent starts in FOOD mode." in desc
    desc = desc.replace(" The agent starts in FOOD mode.", "")
    return desc


def infer(rng, model, chm, num_samples=1000, num_particles=100):
    # https://genjax.gen.dev/cookbook/cookbook/inference/importance_sampling.html
    def SIR(N, K, model, chm):
        @jax.jit
        def _inner(key, args):
            key, subkey = jax.random.split(key)
            traces, weights = jax.vmap(model.importance, in_axes=(0, None, None))(
                jax.random.split(key, N), chm, args
            )
            idxs = jax.vmap(jax.jit(genjax.categorical.simulate), in_axes=(0, None))(
                jax.random.split(subkey, K), (weights,)
            ).get_retval()
            samples = traces.get_sample()
            resampled_samples = jax.vmap(lambda idx: jtu.tree_map(lambda v: v[idx], samples))(
                idxs
            )
            return resampled_samples

        return _inner
    
    samples = jax.jit(SIR(num_samples, num_particles, model, chm))(rng, ())
    return samples



def test_agents(rng, code_blocks, test_env_id, num_steps=5, working_dir=None):
    """Test if all agents in the query output can be simulated without errors.
    
    Args:
        rng: JAX random key
        query_output: Dictionary containing the agents and their parameters
        env: Environment instance
        num_steps: Number of simulation steps (default: 10)
        working_dir: Directory for the interpreter to use (default: None)
        
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    if working_dir is None:
        working_dir = Path(__file__).parent / "results" / Path(__file__).stem / "interpreter"
    working_dir.mkdir(parents=True, exist_ok=True)
    interpreter = Interpreter(working_dir=working_dir)

    # Find all agent IDs in the output
    agent_ids = [k for k in code_blocks if k.startswith("agent/")]

    for agent_id in agent_ids:
        print(f"Testing {agent_id}")
        agent_params_code = code_blocks[f"params/{agent_id}"]
        agent_code = code_blocks[agent_id]
        # Create test code that initializes and simulates the agent
        test_code = f"""
import jax
from llm_genjax import make
from llm_genjax.envs import simulate

{agent_params_code}
{agent_code}

rng = jax.random.PRNGKey(0)
agent_to_test = agent(params())
env = make("{test_env_id}")(params=make(f"params/{test_env_id}")())
simulate(rng, env, agent_to_test, num_steps={num_steps})
"""
        # Execute the test code
        result = interpreter.run(test_code)
        
        if result.exc_type is not None:
            interpreter.cleanup_session()

            error_message = f"Error: {result.term_out}\nTest code: {test_code}"
            return False, error_message
    
    interpreter.cleanup_session()
    return True, None


def query_until_success(rng, prompt, model_id, env_id, save_dir, max_tries=3):
    """Query the LLM repeatedly until a working agent implementation is found.
    
    Args:
        rng: JAX random key
        prompt: Initial prompt for the LLM
        model_id: ID of the LLM model to use
        env_id: Environment ID for testing
        save_dir: Directory to save results
        max_tries: Maximum number of attempts (default: 3)
        
    Returns:
        tuple: (res: dict, code: dict) containing the successful implementation
    """
    res = {}
    code = {}
    messages = []
    success = False
    msg = prompt
    
    for i in range(max_tries):
        _save_dir = save_dir / f"step_{i}"
        _code, _res = query(messages, msg, model_id, _save_dir)
        res.update(_res)
        code.update(_code)

        success, error_message = test_agents(rng, code, env_id)
        if success:
            break
        else:
            (_save_dir / "error.txt").write_text(error_message)
            msg = error_message
            
    return res, code


if __name__ == "__main__":
    num_steps = 3000
    video_steps = 1000
    # video_steps = 10
    env_id = "env/foraging"
    model_id = "claude-3-5-sonnet-latest"

    # gt_agent_id = "agent/nearest_food-sleep"
    gt_agent_id = "agent/nearest_food"

    desc_ids = {
        "agent/nearest_food": [
            "description/agent/nearest_food",
        ],
        "agent/nearest_food-sleep": [
            "description/agent/nearest_food-sleep",
            "description/agent/nearest_food-sleep-variant1",
        ],
    }

    run_names = ["run1"]

    rng = jax.random.PRNGKey(0)

    env = make(env_id)(params=make(f"params/{env_id}")())
    gt_agent = make(gt_agent_id)(params=make(f"params/{gt_agent_id}")())

    # simulate groundtruth agent
    rng, _rng = jax.random.split(rng)
    tr, states, actions, _ = simulate(_rng, env, gt_agent, num_steps=num_steps, video_steps=video_steps, animate=True, save_path=save_dir / "gt" / "animation.gif")


    # get choicemaps
    choices = tr.get_choices()

    # agent positions
    agent_pos_chm = C["sim", :, "env_state", "agent_pos", :].set(states.agent_pos)

    # init food position and sampled food positions
    env_chm = C["init_env_state", "init_food", :, "pos"].set(choices["init_env_state", "init_food", :, "pos"])
    # TODO: does it work if don't add new food pos to choicemap?
    env_chm = env_chm ^ C["sim", :, "env_state", "new_food_pos", "pos"].set(choices["sim", :, "env_state", "new_food_pos", "pos"])

    # for inference, observe both the agent position and the food positions
    inference_chm = agent_pos_chm ^ env_chm

    # sanity check
    rng, _rng = jax.random.split(rng)
    gt_agent2 = make(gt_agent_id)(params=make(f"params/{gt_agent_id}")())
    tr, states, actions, _ = simulate(_rng, env, gt_agent2, num_steps=num_steps, choice_map=inference_chm, animate=True, video_steps=video_steps, save_path=save_dir / "gt" / "animation2.gif")

    for run_name in run_names:
        run_save_dir = save_dir / f"{gt_agent_id}_{run_name}"

        for desc_id in desc_ids[gt_agent_id]:
            _save_dir = run_save_dir / f"desc_{desc_id.split('/')[-1]}"
            gen_dir = _save_dir / "gen"

            prompt = instructions.format(
                env_interface=env_interface,
                genjax_docs=genjax_docs,
                agent_description=make(desc_id)(),
                agent_code=agent_code,
            )

            res, code = query_until_success(rng, prompt, model_id, env_id, gen_dir)
            
            agent_id = list([k for k in res if k.startswith("agent/")])[0]
            agent = res[agent_id](params=res[f"params/{agent_id}"]())

            agents = {k: v(res[f"params/{k}"]()) for k, v in res.items() if k.startswith("agent/")}
            eval_agents(rng, env, gt_agent, agents, num_steps, fps=30, save_dir=gen_dir)


            tr, states, actions, _ = simulate(rng, env, agent, num_steps, video_steps=video_steps, animate=True, save_path=gen_dir / "animation.gif")

            # inference
            rng, _rng = jax.random.split(rng)
            model = make_model(env, agent, num_steps)
            samples = infer(_rng, model, inference_chm, num_samples=10000, num_particles=1000)

            inferred_chm = ChoiceMap.empty()
            inferred_vars = {}
            for k in samples("init_agent_state").mapping.keys():
                print(f"Inferred {k}: {samples['init_agent_state', k].mean(axis=0)}")
                v = samples["init_agent_state", k].mean(axis=0)
                inferred_chm = inferred_chm ^ C["init_agent_state", k].set(v)
                inferred_vars[k] = v

            _chm = env_chm ^ inferred_chm

            # simulate with inferred values
            rng, _rng = jax.random.split(rng)
            tr, states, actions, _ = simulate(_rng, env, agent, choice_map=_chm, num_steps=num_steps, video_steps=video_steps, animate=True, save_path=_save_dir / "inferred" / "animation.gif")

            fitted_agent = res[agent_id](params=res[f"params/{agent_id}"]())
            agents.update({agent_id + "_inferred": fitted_agent})
            choice_maps = {agent_id + "_inferred": _chm}
            # eval
            eval_agents(
                rng,
                env,
                gt_agent,
                agents,
                num_steps,
                fps=30,
                save_dir=_save_dir / "inferred",
                choice_maps=choice_maps,
            )
            

            plt.figure(figsize=(4, 4), dpi=150)
            for k, v in inferred_vars.items():
                gt_value = getattr(gt_agent.params, k)
                plt.scatter(gt_value, v, label=f"{k}", marker="o")
            # identity line
            plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), ls="--", c=".3", zorder=0)
            plt.xlabel("Ground truth")
            plt.ylabel("Inferred")
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend()
            plt.tight_layout()
            plt.savefig(_save_dir / "inferred" / "params.png")

            plt.close("all")
            