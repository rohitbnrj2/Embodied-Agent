import numpy as np
import math
import pygame

from cambrian.utils.pygame_colors import colors as PY_GAME_COLORS

_EPSILON_ = 1e-7

def visualize_rays(display, photoreceptors, visualize_aperture=True):

    for out in photoreceptors:
        rays = out.rays
        photoreceptor_intensity = out.raw_radiance
        for ray in rays: 
            rgb = (ray.intensity * 255).astype(np.uint8)
            # import pdb; pdb.set_trace()
            # ray.type = 'aperture'
            # if ray.type == 'aperture':
            if ray.collision_point is not None: 
                # only render if ray collides with smthing
                if False:
                    if visualize_aperture:
                        pygame.draw.line(display, rgb, (ray.x, ray.y), render_dict[k]['intersection_points'][i])
                else:
                    pygame.draw.line(display, rgb, (ray.x, ray.y), ray.collision_point)
    return display

def ray_wall_collision(ray, walls):
    closest = 100000
    closestPoint = None
    ray_color = None
    for wall in walls:
        ret = ray.checkCollision(wall)
        if ret is not None:
            intersectPoint, wall = ret
            # Get distance between ray source and intersect point
            ray_dx = ray.x - intersectPoint[0]
            ray_dy = ray.y - intersectPoint[1]
            # If the intersect point is closer than the previous closest intersect point, it becomes the closest intersect point
            distance = math.sqrt(ray_dx**2 + ray_dy**2)
            if (distance < closest):
                closest = distance
                closestPoint = intersectPoint
                ray_color = PY_GAME_COLORS[wall.color]

    if closestPoint is not None:
        return closest, closestPoint, ray_color
    return None

def scale_intensity(intensity, max_rays_per_pixel, min_offset=50, scale=1.0):
    # intensity = np.clip(intensity, min_offset, max_rays_per_pixel)
    # return map_one_range_to_other(intensity, 0, 1, min_offset, max_rays_per_pixel)
    return np.clip((intensity/max_rays_per_pixel)*scale, 0., 1.0)

def map_one_range_to_other(input, output_start, output_end, input_start, input_end):
    " end is inclusive! "
    output = output_start + ((output_end - output_start) / (_EPSILON_ + input_end - input_start)) * (input - input_start)
    return output

def touint8(c):
    return map_one_range_to_other(c, 0, 255-1, 0, 1).astype(np.uint8)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def points_on_circumference(center = (0.0, 0.0), r: float = 50.0, n: int = 100, direction='right'):
    """
    Outputs in increasing order of y, a[0] is the lowest value
    """
    pts = []
    b1 = int(-n/2)
    b2 = int(n/2) + 1
    sgn = 1.0
    
    if direction == 'left':
        sgn = -1.0
     
    for x in range(b1, b2):
        _x = 1.0 * center[0] + (sgn * math.cos(2 * math.pi/2 / n * x) * r)  # x
        _y = 1.0 * center[1] + (1.0 * math.sin(2 * math.pi/2 / n * x) * r)  # y
        pts.append([_x, _y])
    return pts

def name_to_fdir(dir):
    if dir == 'left':
        return 1.0
    elif dir == 'right':
        return -1.0
    else:
        ValueError("{} not found".format(dir)) 