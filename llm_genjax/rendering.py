from functools import partial
from enum import IntEnum, Enum, auto

import numpy as np
import jax
import jax.numpy as jnp


class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    ORANGE = auto()


ColorRGB = {
    Color.RED: jnp.array([255, 0, 0]),
    Color.GREEN: jnp.array([0, 255, 0]),
    Color.BLUE: jnp.array([0, 0, 255]),
    Color.ORANGE: jnp.array([255, 165, 0])
}


class Shape(Enum):
    SQUARE = auto()
    CIRCLE = auto()
    TRIANGLE = auto()


# class Frame(Enum):
class Frame(IntEnum):
    WORLD = auto()
    SCREEN_PIXEL = auto()
    WORLD_RELATIVE = auto()
    SCREEN_PIXEL_RELATIVE = auto()
    WORLD_LOCAL_RELATIVE = auto()
    ARRAY_INDEX = auto()


def world_relative_to_array_index(coord, params):
    # (0, 0) -> row=screen_height/2, col=screen_width/2
    # (-1, -1) -> row=0, col=0
    # (1, 1) -> row=screen_height, col=screen_width
    screen_width, screen_height = params.screen_size

    col = (coord[0] + 1) * screen_width / 2
    row = (-coord[1] + 1) * screen_height / 2

    return jnp.array([row, col], dtype=int)


def world_to_array_index(coord, params):
    # (0, 0) -> row=screen_height/2, col=screen_width/2
    # (-W/2, -H/2) -> row=0, col=0
    # (W/2, H/2) -> row=screen_height, col=screen_width
    world_width, world_height = params.world_size
    screen_width, screen_height = params.screen_size

    col = (coord[0]/(world_width/2) + 1) * screen_width / 2
    row = (-coord[1]/(world_height/2) + 1) * screen_height / 2
    return jnp.array([row, col], dtype=int)


@partial(jax.jit, static_argnames=("params",))
def coord_to_array_index(coord, params):
    if params.frame == Frame.ARRAY_INDEX:
        return coord
    elif params.frame == Frame.WORLD:
        return world_to_array_index(coord, params)
    elif params.frame == Frame.WORLD_RELATIVE:
        return world_relative_to_array_index(coord, params)
    elif params.frame == Frame.SCREEN_PIXEL:
        # swap x, y
        return jnp.array([coord[1], coord[0]], dtype=int)
    else:
        raise ValueError(f"Unsupported coordinate frame: {params.frame}")


def size_to_array_index(size, params):
    if params.frame == Frame.ARRAY_INDEX:
        return size
    elif params.frame == Frame.WORLD:
        world_width, world_height = params.world_size
        screen_width, screen_height = params.screen_size
        # return tuple (jit requires fixed sizes)
        size = size[0]*screen_height/world_height, size[1]*screen_width/world_width
        size = int(size[0]), int(size[1])
        return size
    elif params.frame == Frame.WORLD_RELATIVE:
        screen_width, screen_height = params.screen_size
        size = size[0]*screen_height/2, size[1]*screen_width/2
        size = int(size[0]), int(size[1])
        return size
    else:
        raise ValueError(f"Unsupported coordinate frame: {params.frame}")



@partial(jax.jit, static_argnames=("size", "params"))
def draw_square(img, position, size, color, params):
    idx = coord_to_array_index(position, params)
    # TODO: size in render is in screen coords, change it to world coords
    # size = size_to_array_index(size, params)
    size = size_to_array_index(size, params.replace(frame=Frame.WORLD))
    # # center
    idx = idx - jnp.array(size, dtype=int) // 2

    square_idx = jnp.array([idx[0], idx[1], 0], dtype=int)
    square = jnp.ones((size[0], size[1], 3), dtype=img.dtype) * jnp.array(color, dtype=img.dtype)
    # place square on the image at index square_idx
    img = jax.lax.dynamic_update_slice(img, square, square_idx)
    return img


@partial(jax.jit, static_argnames=("size", "params"))
def draw_box(img, position, size, color, params=None):
    row, col = coord_to_array_index(position, params)
    size = size_to_array_index(size, params.replace(frame=Frame.WORLD))
    w, h = size
    # center
    row = row - h // 2
    col = col - w // 2

    bottom_border = jnp.ones((1, size[0], 3), dtype=img.dtype) * color
    top_border = jnp.ones((1, size[0], 3), dtype=img.dtype) * color
    left_border = jnp.ones((size[1], 1, 3), dtype=img.dtype) * color
    right_border = jnp.ones((size[1], 1, 3), dtype=img.dtype) * color

    img = jax.lax.dynamic_update_slice(img, bottom_border, (row + h - 1, col, 0))
    img = jax.lax.dynamic_update_slice(img, top_border, (row, col, 0))
    img = jax.lax.dynamic_update_slice(img, left_border, (row, col, 0))
    img = jax.lax.dynamic_update_slice(img, right_border, (row, col + w - 1, 0))

    return img


@partial(jax.jit, static_argnames=("diameter", "img_size", "params"))
def draw_disk(img, position, diameter, color, img_size, params):
    center = coord_to_array_index(position, params)
    diameter, _ = size_to_array_index((diameter, diameter), params.replace(frame=Frame.WORLD))

    # draw a circle around the center with the given radius and color
    y, x = jnp.meshgrid(jnp.arange(img_size[0]), jnp.arange(img_size[1]))

    radius = diameter / 2
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    img = jax.lax.dynamic_update_slice(img, jnp.where(mask[:, :, None], color, img), (0, 0, 0))
    return img


@partial(jax.jit, static_argnames=("diameter", "img_size", "params"))
def draw_circle(img, position, diameter, color, img_size, border_width=1, params=None):
    center = coord_to_array_index(position, params)
    diameter, _ = size_to_array_index((diameter, diameter), params.replace(frame=Frame.WORLD))

    # draw a circle around the center with the given radius and color
    y, x = jnp.meshgrid(jnp.arange(img_size[0]), jnp.arange(img_size[1]))

    radius = diameter / 2
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    mask = jnp.logical_and(mask, (x - center[0]) ** 2 + (y - center[1]) ** 2 >= (radius - border_width) ** 2)
    img = jax.lax.dynamic_update_slice(img, jnp.where(mask[:, :, None], color, img), (0, 0, 0))
    return img


@partial(jax.jit, static_argnames=("size", "params"))
def draw_triangle(img, position, size, color, params):
    idx = coord_to_array_index(position, params)
    size = size_to_array_index(size, params.replace(frame=Frame.WORLD))

    def inside_triangle(x, y, x1, y1, x2, y2, x3, y3):
        d1 = (x - x2) * (y1 - y2) - (y - y2) * (x1 - x2)
        d2 = (x - x3) * (y2 - y3) - (y - y3) * (x2 - x3)
        d3 = (x - x1) * (y3 - y1) - (y - y1) * (x3 - x1)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_neg & has_pos)

    # Define the vertices of the triangle
    p1 = jnp.array([idx[0] - size[1] / 2, idx[1]])
    p2 = jnp.array([idx[0] + size[1] / 2, idx[1] - size[0] / 2])
    p3 = jnp.array([idx[0] + size[1] / 2, idx[1] + size[0] / 2])

    # Create a meshgrid of coordinates
    yy, xx = jnp.meshgrid(jnp.arange(img.shape[1]), jnp.arange(img.shape[0]))

    # Determine which pixels are inside the triangle
    mask = inside_triangle(xx, yy, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    mask = mask[:, :, None]

    # Create the triangle with the specified color
    triangle = jnp.ones_like(img) * jnp.array(color, dtype=img.dtype)
    triangle = jnp.where(mask, triangle, img)

    return triangle


@partial(jax.jit, static_argnames=("size", "params"))
def draw_triangle_outline(img, position, size, color, border_width=1, params=None):
    idx = coord_to_array_index(position, params)
    size = size_to_array_index(size, params.replace(frame=Frame.WORLD))

    def inside_triangle(x, y, x1, y1, x2, y2, x3, y3):
        d1 = (x - x2) * (y1 - y2) - (y - y2) * (x1 - x2)
        d2 = (x - x3) * (y2 - y3) - (y - y3) * (x2 - x3)
        d3 = (x - x1) * (y3 - y1) - (y - y1) * (x3 - x1)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_neg & has_pos)

    def edge_distance(x, y, x1, y1, x2, y2):
        return jnp.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / jnp.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Define the vertices of the triangle
    p1 = jnp.array([idx[0] - size[1] / 2, idx[1]])
    p2 = jnp.array([idx[0] + size[1] / 2, idx[1] - size[0] / 2])
    p3 = jnp.array([idx[0] + size[1] / 2, idx[1] + size[0] / 2])
    # shift points up
    # TODO: ad hoc value
    shift = jnp.array([-2, 0])
    p1 += shift
    p2 += shift
    p3 += shift
    # Create a meshgrid of coordinates
    yy, xx = jnp.meshgrid(jnp.arange(img.shape[1]), jnp.arange(img.shape[0]))

    # Determine which pixels are inside the triangle
    mask = inside_triangle(xx, yy, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

    # Determine which pixels are close to the edges
    edge_mask1 = edge_distance(xx, yy, p1[0], p1[1], p2[0], p2[1]) < border_width
    edge_mask2 = edge_distance(xx, yy, p2[0], p2[1], p3[0], p3[1]) < border_width
    edge_mask3 = edge_distance(xx, yy, p3[0], p3[1], p1[0], p1[1]) < border_width

    edge_mask = jnp.logical_or(jnp.logical_or(edge_mask1, edge_mask2), edge_mask3)
    outline_mask = jnp.logical_and(mask, edge_mask)
    outline_mask = outline_mask[:, :, None]

    # Create the triangle outline with the specified color
    triangle_outline = jnp.ones_like(img) * jnp.array(color, dtype=img.dtype)
    img = jax.lax.dynamic_update_slice(img, jnp.where(outline_mask, triangle_outline, img), (0, 0, 0))

    return img


# TODO: set shape to be static so that can have if else in the function
@partial(jax.jit, static_argnames=("size", "params"))
def draw_shape(img, shape, position, size, color, params=None):
    img_square = draw_square(img, position, size, color, params=params)
    img_circle = draw_disk(img, position, size[0], color, img_size=img.shape[:2], params=params)
    img_triangle = draw_triangle(img, position, size, color, params=params)

    img = jax.lax.switch(
        shape,
        [
            lambda: img_square,
            lambda: img_circle,
            lambda: img_triangle
        ]
    )
    return img


@partial(jax.jit, static_argnames=("size", "params"))
def draw_shape_outline(img, shape, position, size, color, border_width=1, params=None):
    img_square = draw_box(img, position, size, color, params=params)
    img_circle = draw_circle(img, position, size[0], color, border_width=border_width, img_size=img.shape[:2], params=params)
    img_triangle = draw_triangle_outline(img, position, size, color, params=params)

    img = jax.lax.switch(
        shape,
        [
            lambda: img_square,
            lambda: img_circle,
            lambda: img_triangle
        ]
    )
    return img


@partial(jax.jit, static_argnames=("dot_spacing", "dot_size", "dot_color"))
def draw_grid(img, dot_spacing=20, dot_size=2, dot_color=(200, 200, 200)):
    height, width, _ = img.shape
    # dot_size x dot_size dot array
    dot = jnp.ones((dot_size, dot_size, 3), dtype=img.dtype) * jnp.array(dot_color, dtype=img.dtype)
    rows = jnp.arange(0, height, dot_spacing)
    cols = jnp.arange(0, width, dot_spacing)

    def draw_dots(img, row):
        def draw_single_dot(img, col):
            # use dynamic slicing
            return jax.lax.dynamic_update_slice(img, dot, (row, col, 0)), _
        return jax.lax.scan(draw_single_dot, img, cols)[0], _
    img, _ = jax.lax.scan(draw_dots, img, rows)
    return img


@partial(jax.jit, static_argnames=("size", "params"))
def draw_bar(img, level, position, size, color, params):
    # drawing a dynamic bar with jax is little tricky since array shapes have to be known at compile time (cannot depend on values of other arrays)
    img_size = jnp.array(img.shape[:2])

    x, y = position
    x = x.astype(jnp.int32)
    y_bottom = (img_size[1] - y - size[1]).astype(jnp.int32)
    y_top = (img_size[1] - y).astype(jnp.int32)

    # mask for the bar, points above level on screen are transparent
    # row=0, col=0 of img corresponds to top left corner of the screen (pygame convention)
    mask = jnp.arange(size[1]) > jnp.round(size[1] - level * size[1])
    full_bar = jnp.ones((size[1], size[0], 3), dtype=img.dtype) * jnp.array(color, dtype=img.dtype)
    bar = jnp.where(
        mask[:, None, None],
        full_bar,
        jax.lax.dynamic_slice(img, (y_bottom, x, 0), (size[1], size[0], 3))
    )
    img = jax.lax.dynamic_update_slice(img, bar, (y_bottom, x, 0))

    # black border around the bar
    outline_color = jnp.array([0, 0, 0], dtype=img.dtype)
    params = params.replace(frame=Frame.SCREEN_PIXEL)
    # TODO: issue with size
    # img = draw_box(img, (x+size[0]/2, y_bottom+size[1]/2), size, outline_color, params=params)

    # TODO: temp fix
    color = outline_color
    row, col = coord_to_array_index((x+size[0]/2, y_bottom+size[1]/2), params)
    w, h = size
    # center
    row = row - h // 2
    col = col - w // 2

    bottom_border = jnp.ones((1, size[0], 3), dtype=img.dtype) * color
    top_border = jnp.ones((1, size[0], 3), dtype=img.dtype) * color
    left_border = jnp.ones((size[1], 1, 3), dtype=img.dtype) * color
    right_border = jnp.ones((size[1], 1, 3), dtype=img.dtype) * color

    img = jax.lax.dynamic_update_slice(img, bottom_border, (row + h - 1, col, 0))
    img = jax.lax.dynamic_update_slice(img, top_border, (row, col, 0))
    img = jax.lax.dynamic_update_slice(img, left_border, (row, col, 0))
    img = jax.lax.dynamic_update_slice(img, right_border, (row, col + w - 1, 0))

    return img


@partial(jax.jit, static_argnames=("size", "params"))
def render_food_sources(img, state, size, params):
    outline_size = (15, 15)
    outline_width = 1

    def _render_food_source(carry, food_source):
        img = carry
        food_color = jnp.array(food_source.color)
        # white background
        img = draw_shape(img, food_source.shape, food_source.position, size=outline_size, color=jnp.array([255, 255, 255]), params=params)
        # draw food shape if food source is not empty
        img_shape = draw_shape(img, food_source.shape, food_source.position, size=size, color=food_color, params=params)
        img = jax.lax.select(
            food_source.empty,
            img,
            img_shape
        )
        # draw outline
        img = draw_shape_outline(img, food_source.shape, food_source.position, size=outline_size, color=food_color, border_width=outline_width, params=params)
        return img, None

    food_sources = state.food_sources
    img, _ = jax.lax.scan(_render_food_source, img, food_sources)
    return img


def render_partial_view(img, agent_position, params):
    screen_width, screen_height = params.screen_size
    view_width, view_height = params.view_size
    # pad world to get observation when agent is near the edge of the world
    padded_world = jnp.ones((screen_width + view_width, screen_height + view_height, 3), dtype=np.uint8) * 255
    padded_world = padded_world.at[view_height//2:view_height//2 + screen_height, view_width//2:view_width//2 + screen_width].set(img)

    agent_coord = coord_to_array_index(agent_position, params)
    img = jax.lax.dynamic_slice(
        padded_world,
        (agent_coord[0], agent_coord[1], 0),
        (view_height, view_width, 3)
    )
    return img





@partial(jax.jit, static_argnames=("view_distance", "params"))
def draw_fov(img, position, heading, fov_angle, view_distance, color=(200, 200, 200, 128), params=None):
    """Draw field of view as a semi-transparent arc.
    
    Args:
        img: Image array (H,W,3)
        position: Agent position (x,y)
        heading: Agent heading vector (d1,d2)
        fov_angle: Field of view angle in radians
        view_distance: Maximum view distance
        color: RGBA color tuple for FOV visualization
        params: Rendering parameters
    """
    # Convert position to array coordinates
    center = coord_to_array_index(position, params)
    
    view_distance_screen = size_to_array_index([view_distance, view_distance], params)[0]

    # Calculate heading angle
    heading_angle = jnp.arctan2(heading[1], heading[0])
    
    # Debug: Add markers for heading direction and FOV edges
    # # Forward point
    # forward_pos = (
    #     position[0] + view_distance * jnp.cos(heading_angle),
    #     position[1] + view_distance * jnp.sin(heading_angle)
    # )
    # img = draw_square(img, forward_pos, (5, 5), (255, 0, 0), params)  # Red marker
    
    # # Left edge of FOV
    # left_angle = heading_angle - fov_angle/2
    # left_pos = (
    #     position[0] + view_distance * jnp.cos(left_angle),
    #     position[1] + view_distance * jnp.sin(left_angle)
    # )
    # img = draw_square(img, left_pos, (5, 5), (0, 255, 0), params)  # Green marker
    
    # # Right edge of FOV
    # right_angle = heading_angle + fov_angle/2
    # right_pos = (
    #     position[0] + view_distance * jnp.cos(right_angle),
    #     position[1] + view_distance * jnp.sin(right_angle)
    # )
    # img = draw_square(img, right_pos, (5, 5), (0, 0, 255), params)  # Blue marker
    
    # Create meshgrid for the entire image
    x, y = jnp.meshgrid(jnp.arange(img.shape[0]), jnp.arange(img.shape[1]))

    # Calculate angle and distance for each pixel relative to agent
    dx = x - center[0]
    dy = y - center[1]
    distances = jnp.sqrt(dx**2 + dy**2)
    angles = jnp.arctan2(dy, dx)
    
    # Normalize angles relative to heading
    rel_angles = (angles - heading_angle + jnp.pi/2) % (2 * jnp.pi) - jnp.pi
    
    # Create FOV mask
    fov_mask = (jnp.abs(rel_angles) <= fov_angle/2) & (distances <= view_distance_screen)
    fov_mask = fov_mask.T[:, :, None]  # Add channel dimension and transpose
    
    # Create FOV overlay
    fov_overlay = jnp.ones_like(img) * jnp.array(color[:3], dtype=img.dtype)
    
    # Blend FOV overlay with original image using alpha
    alpha = color[3] / 255.0
    img = jnp.where(
        fov_mask,
        (1 - alpha) * img + alpha * fov_overlay,
        img
    )

    return img.astype(jnp.uint8)
