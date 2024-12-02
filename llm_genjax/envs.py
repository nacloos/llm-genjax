import enum
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax
import jax.numpy as jnp
from flax import struct

import genjax
from genjax import gen

from llm_genjax import make, register, registry
from llm_genjax.rendering import (
    Frame,
    draw_square,
    draw_bar,
    draw_fov,
)

@struct.dataclass
class Env:
    init: genjax.StaticGenerativeFunction
    step: genjax.StaticGenerativeFunction
    params: dataclass
    render: Callable


@struct.dataclass
class Agent:
    init: genjax.StaticGenerativeFunction
    step: genjax.StaticGenerativeFunction
    params: dataclass



@register("params/env/foraging")
@struct.dataclass
class ParamsEnvForaging:
    # physics
    friction: float = 0.9
    mass: float = 100.

    # displacement
    max_step_size: float = 2.5
    # max_velocity: float = 0.03
    angular_step_size: float = 0.2

    world_size: tuple = (400, 400)
    screen_size: tuple = (400, 400)
    food_zone_size: tuple = (300, 300)

    # Food color values
    # food_color_values: tuple[float, float] = (-0.1, 0.1)
    food_color_values: tuple[float, float] = (0.2, 0.2)

    sleep_decay: float = 0.09
    sleep_increment: float = 0.13
    sleep_max: float = 1.0
    sleep_threshold: float = 0.3

    food_decay: float = 0.002  # food percentage per second
    food_max: float = 1.
    food_threshold: float = 0.3
    num_food_sources: int = 5

    good_food_prob: float = 0.5  # assume good food = idx 1

    # if food value (or estimated q value) is less than this threshold, avoid
    # avoid_threshold: float = -0.5
    avoid_threshold: float = -0.05
    learning_rate: float = 0.1

    # food_sample_radius: float = 250.0
    # food_sample_spread: float = 40.0
    food_eating_threshold: float = 5.0
    food_near_threshold: float = 15.0
    nest_threshold: float = 10.0
    agent_pos_noise: float = 0.1

    # field of view
    fov_angle: float = 2.0
    view_distance: float = 50.0
    local_view: bool = False

    # rendering
    fps: int = 30
    frame: Frame = Frame.WORLD  # coordinate system
    render_partial_view: bool = True
    nest_size: tuple[int, int] = (25, 25)
    nest_color: tuple[int, int, int] = (255, 165, 0)
    agent_size: tuple[int, int] = (10, 10)
    agent_color: tuple[int, int, int] = (0, 0, 0)
    food_source_size: tuple[int, int] = (8, 8)
    food_colors: tuple = (
        (0, 255, 0),
        (135, 206, 235),
        (255, 99, 71),
    )

    bar_size: tuple[int, int] = (10, 50)
    sleep_bar_color: tuple[int, int, int] = (255, 165, 0)
    food_bar_color: tuple[int, int, int] = (0, 255, 0)
    sleep_bar_pos: tuple[int, int] = (10, 10)
    food_bar_pos: tuple[int, int] = (25, 10)



@register("env/foraging")
def env_foraging(params):
    class BehaviorMode(enum.IntEnum):
        REST = 0
        EXPLORE = 1
        FOOD = 2  # active when sense food

    @struct.dataclass
    class StateForaging:
        # agent state
        agent_pos: jax.Array
        agent_vel: jax.Array
        # agent_theta: float
        # agent_step_size: float
        agent_food_level: float
        agent_sleep_level: float
        agent_mode: BehaviorMode  # current mode of behavior
        agent_food_q_values: jax.Array

        eating_food: bool
        in_nest: bool

        prev_agent_pos: jax.Array
        prev_pos_error: jax.Array
        food_positions: jax.Array
        food_colors: jax.Array
        food_color_values: jax.Array
        nest_pos: jax.Array

    # @gen
    # def sample_food_position():
    #     # Use params instead of hardcoded values
    #     radius = genjax.uniform(params.food_sample_radius, params.food_sample_radius) @ "radius"
    #     angle = genjax.uniform(0.0, 2 * jnp.pi) @ "angle"
    #     base_x = radius * jnp.cos(angle)
    #     base_y = radius * jnp.sin(angle)

    #     x = genjax.normal(base_x, params.food_sample_spread) @ "x"
    #     y = genjax.normal(base_y, params.food_sample_spread) @ "y"
    #     return jnp.array([x, y])

    # sample food uniformly in world
    @gen
    def sample_food_position():
        return genjax.uniform(jnp.array([
            -params.food_zone_size[0]/2,
            -params.food_zone_size[1]/2,
        ]), jnp.array([
            params.food_zone_size[0]/2,
            params.food_zone_size[1]/2,
        ])) @ "pos"

    # @gen
    # def sample_food_color():
        # return genjax.bernoulli(params.good_food_prob) @ "color"

    def food_near(state, threshold=params.food_near_threshold):
        distances = jnp.linalg.norm(state.food_positions - state.agent_pos, axis=1)
        # eat food is close enough and in foraging mode
        is_food_near = jnp.any(distances < threshold)
        # find the nearest food source
        food_idx = jnp.argmin(distances)
        food_pos = state.food_positions[food_idx]
        return is_food_near, food_idx, food_pos

    def update_food_level(state, is_food_eaten, food_value):
        # update agent food level
        agent_food_level = state.agent_food_level - params.food_decay
        agent_food_level = jax.lax.select(
            is_food_eaten,
            agent_food_level + food_value,
            agent_food_level
        )
        agent_food_level = jnp.clip(agent_food_level, 0.0, params.food_max)
        state = state.replace(
            agent_food_level=agent_food_level
        )
        return state

    def update_sleep_level(state):
        # Use params.nest_threshold instead of hardcoded 10.0
        agent_in_nest = jnp.all(jnp.abs(state.agent_pos - state.nest_pos) <= params.nest_threshold)

        agent_sleep_level = jax.lax.select(
            agent_in_nest,
            state.agent_sleep_level + params.sleep_increment / params.fps,
            state.agent_sleep_level - params.sleep_decay / params.fps
        )

        # clip value
        agent_sleep_level = jnp.clip(agent_sleep_level, 0.0, params.sleep_max)

        state = state.replace(
            agent_sleep_level=agent_sleep_level,
            in_nest=agent_in_nest,
        )
        return state

    def update_food_q_values(state, is_food_eaten, food_value, food_color, learning_rate):
        # update agent food q values
        q_values = state.agent_food_q_values
        q = q_values[food_color]
        new_q = q + learning_rate * (food_value - q)

        q_values = jax.lax.select(
            is_food_eaten,
            q_values.at[food_color].set(new_q),
            q_values,
        )
        state = state.replace(
            agent_food_q_values=q_values
        )
        # print all the relevant values
        # jax.debug.print("is_food_eaten: {is_food_eaten}, food_value: {food_value}, food_color: {food_color}, agent_q_values: {agent_food_q_values}", is_food_eaten=is_food_eaten, food_value=food_value, food_color=food_color, agent_food_q_values=agent_food_q_values)
        return state

    @gen
    def init_state():
        theta = genjax.uniform(0.0, 2 * jnp.pi) @ "init_theta"

        food_positions = genjax.repeat(n=params.num_food_sources)(sample_food_position)() @ "init_food"
        # food_colors = genjax.repeat(n=params.num_food_sources)(sample_food_color)() @ "init_food_colors"

        # all food sources have the same color (= params.food_source_color)
        food_colors = jnp.zeros(params.num_food_sources, dtype=jnp.int32)

        food_color_values = jnp.array(params.food_color_values)

        state = StateForaging(
            agent_pos=jnp.array([0.0, 0.0]),
            agent_vel=jnp.array([0.0, 0.0]),
            # agent_theta=theta,
            # agent_step_size=params.max_step_size,
            agent_food_level=1.0,
            agent_sleep_level=1.0,
            agent_mode=BehaviorMode.EXPLORE,
            agent_food_q_values=jnp.ones(2, dtype=jnp.float32)*0.0,
            eating_food=jnp.asarray(False, dtype=jnp.bool_),
            in_nest=jnp.asarray(True, dtype=jnp.bool_),

            prev_pos_error=jnp.zeros(2, dtype=jnp.float32),
            prev_agent_pos=jnp.zeros(2, dtype=jnp.float32),
            food_positions=food_positions,
            food_colors=food_colors,
            food_color_values=food_color_values,
            nest_pos=jnp.array([0.0, 0.0]),
        )
        return state


    @gen
    def step_env(state, action):
        """
        Args:
            state: current state
            action: delta position
            # action: (step_size, theta)
        """
        # Move the agent

        # if action = vel
        # state = state.replace(
        #     prev_agent_pos=state.agent_pos,
        #     agent_pos=state.agent_pos + action,
        #     agent_vel=action,
        #     agent_theta=jnp.arctan2(action[1], action[0]),
        #     agent_step_size=jnp.linalg.norm(action),
        # )

        # step_size = action[0]
        # theta = action[1]
        # delta_pos = step_size * jnp.array([jnp.cos(theta), jnp.sin(theta)])
        delta_pos = action
        state = state.replace(
            # agent_step_size=step_size,
            # agent_theta=theta,
            agent_pos=state.agent_pos + delta_pos,
            agent_vel=delta_pos,
            prev_agent_pos=state.agent_pos,
            eating_food=jnp.asarray(False, dtype=jnp.bool_),
        )

        # Resample the food position if the agent is very close to it (automatic eating)
        is_food_eaten, food_idx, food_pos = food_near(state, threshold=params.food_eating_threshold)
        # is_food_eaten, food_idx, food_pos = sense_food_fov(state, threshold=5)

        new_food_pos = sample_food_position() @ "new_food_pos"
        sampled_food_pos = jax.lax.select(
            is_food_eaten,
            new_food_pos,
            state.food_positions[food_idx],
        )
        state = state.replace(
            food_positions=state.food_positions.at[food_idx].set(sampled_food_pos),
            eating_food=is_food_eaten,
        )

        food_color = state.food_colors[food_idx]
        food_value = state.food_color_values[food_color]
        state = update_food_level(state, is_food_eaten, food_value)
        state = update_sleep_level(state)

        state = update_food_q_values(state, is_food_eaten, food_value, food_color, params.learning_rate)

        # clip pos to be within the world size
        state = state.replace(
            agent_pos=jnp.array(
                [
                    jnp.clip(state.agent_pos[0], -params.world_size[0]/2+params.agent_size[0], params.world_size[0]/2-params.agent_size[0]),
                    jnp.clip(state.agent_pos[1], -params.world_size[1]/2+params.agent_size[1], params.world_size[1]/2-params.agent_size[1]),
                ]
            ),
        )

        agent_pos = genjax.vmap(in_axes=(0, None))(genjax.normal)(state.agent_pos, params.agent_pos_noise) @ "agent_pos"
        state = state.replace(agent_pos=agent_pos)
        return state


    @partial(jax.jit, static_argnums=(1,))
    def render(state, params):
        screen_width, screen_height = params.screen_size
        img = jnp.ones((screen_height, screen_width, 3), dtype=jnp.uint8) * 255

        # render nest
        img = draw_square(img, state.nest_pos, params.nest_size, params.nest_color, params)

        # render agent
        img = draw_square(img, state.agent_pos, params.agent_size, params.agent_color, params)

        # Render all food sources
        def _draw_food(carry, inputs):
            img, idx, food_colors = carry
            food_pos, food_color_idx = inputs
            img = draw_square(img, food_pos, params.food_source_size, food_colors[food_color_idx], params)
            return (img, idx+1, food_colors), None

        food_colors = jnp.array(params.food_colors)
        (img, _, _), _ = jax.lax.scan(_draw_food, (img, 0, food_colors), (state.food_positions, state.food_colors))

        if params.local_view:
            # Draw FOV
            img = draw_fov(
                img, 
                state.agent_pos,
                state.agent_vel,
                params.fov_angle,
                params.view_distance,
                color=(20, 20, 20, 64),  # Light gray with alpha=64
                params=params
            )
            
            # Render agent (on top of FOV)
            img = draw_square(img, state.agent_pos, params.agent_size, params.agent_color, params)

        # render sleep level
        img = draw_bar(
            img,
            state.agent_sleep_level,
            position=params.sleep_bar_pos,
            size=params.bar_size,
            color=params.sleep_bar_color,
            params=params
        )
        # render food level
        img = draw_bar(
            img,
            state.agent_food_level,
            position=params.food_bar_pos,
            size=params.bar_size,
            color=params.food_bar_color,
            params=params
        )
        # render food q values
        # def _draw_food_q_values(carry, inputs):
        #     img, idx, food_colors = carry
        #     q_value = inputs
        
        #     # TODO
        #     # rescale -1, 1 to 0, 1
        #     # q_value = (q_value + 1) / 2
            
        #     # rescale -0.1, 0.1 to 0, 1
        #     q_value = (q_value + 0.1) / 0.2

        #     img = draw_bar(
        #         img,
        #         q_value,
        #         position=params.food_bar_pos + (params.bar_size[0]+5)*(idx+2),  # Offset each bar
        #         size=params.bar_size,
        #         color=food_colors[idx],
        #         params=params
        #     )
        #     return (img, idx+1, food_colors), None

        # (img, _, _), _ = jax.lax.scan(_draw_food_q_values, (img, 0, food_colors), state.agent_food_q_values)
        return img

    return Env(init=init_state, step=step_env, params=params, render=render)


# agent utils
# encapsulate gen model in function so that can take params as input (genjax error if use partial)
def move_to_target_pd(params):
    @gen
    def _move_to_target_pd(agent_state, env_state, target_pos):
        # Error (distance to target)
        error = target_pos - env_state.agent_pos
        prev_error = target_pos - env_state.prev_agent_pos
        
        # Proportional term
        P_term = params.kp * error

        # Derivative term (assuming we store previous error in state)
        D_term = params.kd * (error - prev_error)
        
        # Calculate new velocity
        new_velocity = P_term + D_term
        
        # Limit velocity magnitude to step_size
        velocity_magnitude = jnp.linalg.norm(new_velocity)
        new_velocity = jnp.where(
            velocity_magnitude > params.speed,
            new_velocity * params.speed / velocity_magnitude,
            new_velocity
        )

        # Stop moving if on target
        new_velocity = jax.lax.select(
            jnp.linalg.norm(error) < params.error_threshold,
            jnp.zeros(2),
            new_velocity,
        )
        return new_velocity

        # step_size = jnp.linalg.norm(new_velocity)
        # theta = jnp.arctan2(new_velocity[1], new_velocity[0])
        # action = jnp.array([step_size, theta])
        # return action
    return _move_to_target_pd


def goto_nearest_food(params):
    @gen
    def _goto_nearest_food(agent_state, env_state):
        # Go to the nearest food source
        distances = jnp.linalg.norm(env_state.food_positions - env_state.agent_pos, axis=1)
        food_idx = jnp.argmin(distances)
        food_pos = env_state.food_positions[food_idx]
        action = move_to_target_pd(params)(agent_state, env_state, food_pos) @ "action"
        return action
    return _goto_nearest_food


@register("params/agent/constant_speed")
@struct.dataclass
class ParamsAgentConstantSpeed:
    speed_min: float = 0.0
    speed_max: float = 5.0


@register("agent/constant_speed")
def agent_constant_speed(params):
    @struct.dataclass
    class State:
        speed: float
        direction: jnp.ndarray

    @gen
    def init_state():
        speed = genjax.uniform(params.speed_min, params.speed_max) @ "speed"
        direction = genjax.uniform(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])) @ "direction"
        return State(
            speed=speed,
            direction=direction
        )
    
    @gen
    def step(state, env_state):
        action = state.speed * state.direction
        return state, action

    return Agent(init=init_state, step=step, params=params)


@register("params/agent/nearest_food")
@struct.dataclass
class ParamsAgentNearestFood:
    speed: float = 2.5
    kp: float = 0.5
    kd: float = 0.3
    error_threshold: float = 1.0



@register("agent/nearest_food")
def agent_nearest_food(params):
    @struct.dataclass
    class State:
        speed: float

    @gen
    def init_state():
        return State(
            speed=params.speed,
        )

    @gen
    def step(state, env_state):
        # Go to the nearest food source
        action = goto_nearest_food(params)(state, env_state) @ "action"
        return state, action

    return Agent(init=init_state, step=step, params=params)


@register("params/agent/nearest_food-sleep")
@struct.dataclass
class ParamsAgentNearestFoodSleep:
    speed: float = 2.5
    food_threshold: float = 0.3
    sleep_threshold: float = 0.3

    kp: float = 0.5
    kd: float = 0.3
    error_threshold: float = 1.0


@register("agent/nearest_food-sleep")
def agent_nearest_food_sleep(params):
    class BehaviorMode(enum.IntEnum):
        REST = 0
        FOOD = 1

    @struct.dataclass
    class State:
        mode: BehaviorMode

    @gen
    def init_state():
        return State(
            mode=BehaviorMode.FOOD,
        )

    @gen
    def move_to_target_pd(agent_state, env_state, target_pos, threshold=params.error_threshold):
        # Error (distance to target)
        error = target_pos - env_state.agent_pos
        prev_error = target_pos - env_state.prev_agent_pos
        
        # Proportional term
        P_term = params.kp * error

        # Derivative term (assuming we store previous error in state)
        D_term = params.kd * (error - prev_error)
        
        # Calculate new velocity
        new_velocity = P_term + D_term
        
        # Limit velocity magnitude to step_size
        velocity_magnitude = jnp.linalg.norm(new_velocity)
        new_velocity = jnp.where(
            velocity_magnitude > params.speed,
            new_velocity * params.speed / velocity_magnitude,
            new_velocity
        )

        # Stop moving if on target
        new_velocity = jax.lax.select(
            jnp.linalg.norm(error) < threshold,
            jnp.zeros(2),
            new_velocity,
        )
        return new_velocity

        # step_size = jnp.linalg.norm(new_velocity)
        # theta = jnp.arctan2(new_velocity[1], new_velocity[0])
        # action = jnp.array([step_size, theta])
        # return action

    # TODO: beartype error if use the two agent utils functions defined above (with the params arg)
    # temp fix: copy the code here
    @gen
    def goto_nearest_food(agent_state, env_state):
        # Go to the nearest food source
        distances = jnp.linalg.norm(env_state.food_positions - env_state.agent_pos, axis=1)
        food_idx = jnp.argmin(distances)
        food_pos = env_state.food_positions[food_idx]
        action = move_to_target_pd(agent_state, env_state, food_pos) @ "action"
        return action

    @gen
    def step(state, env_state):
        # update mode
        hungry = env_state.agent_food_level < params.food_threshold
        tired = env_state.agent_sleep_level < params.sleep_threshold
        mode = state.mode
        mode = jax.lax.select(
            # if don't rest and sleep level is low, rest
            (mode != BehaviorMode.REST) & tired,
            BehaviorMode.REST,
            mode,
        )
        mode = jax.lax.select(
            # if rest and food level is low, explore
            (mode == BehaviorMode.REST) & hungry,
            BehaviorMode.FOOD,
            mode,
        )
        mode = jax.lax.select(
            # if both tired and hungry, rest
            hungry & tired,
            BehaviorMode.REST,
            mode,
        )
        state = state.replace(mode=mode)

        # if mode is food, go to nearest food
        # if mode is rest, go to nest
        action = genjax.or_else(
            move_to_target_pd,
            goto_nearest_food,
            # TODO: raise beartype error
            # move_to_target_pd(params),
            # goto_nearest_food(params),
        )(
            mode == BehaviorMode.REST,
            (state, env_state, env_state.nest_pos, 10.0),
            (state, env_state),
        ) @ "action"
        return state, action

    return Agent(init=init_state, step=step, params=params)


def make_model(env, agent, num_steps):
    @gen 
    def simulate():
        @gen
        def step(state, _):
            env_state, agent_state = state

            next_agent_state, action = agent.step(agent_state, env_state) @ "agent_state"
            next_env_state = env.step(env_state, action) @ "env_state"

            return (next_env_state, next_agent_state), (next_env_state, action)

        init_env_state = env.init() @ "init_env_state"
        init_agent_state = agent.init() @ "init_agent_state"
        init_state = (init_env_state, init_agent_state)
        _, out = genjax.scan(n=num_steps)(step)(init_state, None) @ "sim"
        return out

    return simulate



def simulate(rng, env, agent, num_steps, choice_map=None, save_dir=None):
    model = make_model(env, agent, num_steps)

    if choice_map is None:
        tr = jax.jit(model.simulate)(rng, ())
    else:
        tr, weight = jax.jit(model.importance)(rng, choice_map, ())

    states, actions = tr.get_retval()
    return tr, states, actions


def plot_agent_pos(states, save_dir=None):
    # plot agent position over time
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(states.agent_pos[:, 0], label="x")
    plt.plot(states.agent_pos[:, 1], label="y")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "agent_pos.png")



def animate_frames(frames, fps=30, title_frame=False, save_path=None):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    if title_frame:
        plt.title("Frame 0")
    else:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


    def animate(i):
        patch.set_data(frames[i])

        if title_frame:
            plt.title(f"Frame {i}")

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=len(frames) / fps)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="imagemagick", fps=fps)
    else:
        plt.show()

if __name__ == "__main__":
    save_dir = Path(__file__).parent / "results" / Path(__file__).stem

    env_id = "env/foraging"
    save_dir = save_dir / env_id.replace("env/", "")

    env_params = make(f"params/{env_id}")()
    env = make(env_id)(params=env_params)

    rng = jax.random.PRNGKey(0)
    # agent_ids = [k for k in registry if k.startswith("agent/")]
    agent_ids = ["agent/nearest_food-sleep"]
    for agent_id in agent_ids:
        agent_params = make(f"params/{agent_id}")()
        agent = make(agent_id)(params=agent_params)

        num_steps = 1000
        tr, states, actions = simulate(rng, env, agent, num_steps)
        frames = jax.vmap(env.render, in_axes=(0, None))(states, env.params)

        _save_dir = save_dir / agent_id.replace("agent/", "")
        plot_agent_pos(states, save_dir=_save_dir)
        animate_frames(frames, fps=30, title_frame=True, save_path=_save_dir / "animation.gif")
