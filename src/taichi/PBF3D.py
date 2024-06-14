import taichi as ti
import numpy as np
import math
import os

ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)
# ti.init(arch=ti.cpu, debug=True, cpu_max_num_threads=1, advanced_optimization=False) # debug

# -----PARAMETERS-----

# If this is True, then the program will use some non-atomic operations, which may cause some problems
# but will improve the performance a lot.
unsafe = False

# -WORLD-
dt = 1.0 / 20.0
solve_iteration = 10
res = (500, 500)
world = (20, 20, 20)
boundary = 20
dimension = 3

# -Visual-
background_color = ti.Vector([0.5, 0.5, 0.5])
visual_radius = 0.5
particle_color = 0x34ebc6
fluid_filter_color = ti.Vector([1.0, 0.8, 0.1])
fluid_reflection_ratio = 1.5
fluid_attenuation_coefficient = 0.1

# -Fluid_Setting-
num_particles = 12000
mass = 1.0
density = 1.0
rest_density = 1.0
radius = 0.4


# -Neighbours_Setting-
h = 1.0
h_2 = h * h
h_6 = h * h * h * h * h * h
h_9 = h * h * h * h * h * h * h * h * h
max_neighbour = 8000

# -Grid_Setting-
grid_size = 1
grid_rows = int(world[0] / grid_size)
grid_cols = int(world[1] / grid_size)
grid_layers = int(world[2] / grid_size)
max_particle_in_grid = 8000

lnm_grid_size = 1
lnm_grid_rows = int(world[0] / lnm_grid_size)
lnm_grid_cols = int(world[1] / lnm_grid_size)
lnm_grid_layers = int(world[2] / lnm_grid_size)
num_grids = lnm_grid_rows * lnm_grid_cols * lnm_grid_layers

# -Boundary Epsilon-
b_epsilon = 0.01

# -POLY6_KERNEL-
poly6_Coe = 315.0 / (64 * math.pi)

# -SPIKY_KERNEL-
spiky_Coe = -45.0 / math.pi

# -LAMBDAS-
lambda_epsilon = 100.0

# -S_CORR-
S_Corr_delta_q = 0.3
S_Corr_k = 0.0001

# -Confinement/ XSPH Viscosity-
xsph_c = 0.01
vorti_epsilon = 0.01

# -Gradient Approx. delta difference-
g_del = 0.01

TYPE = 2
fov = math.pi / 2

filter_radius = 10
thickness_filter_radius = 10

light_position = ti.Vector([0.0, 0.0, 1.0])
light_angle = math.pi / 12
light_color = ti.Vector([1.0, 1.0, 0.9])
camera_angle = ti.Vector([0.0, 1.0, 0.0])

# -----FIELDS-----
position = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
l1_points = ti.field(dtype=ti.i32, shape=num_particles)
last_position = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
lambdas = ti.field(dtype=ti.f32, shape=num_particles)
delta_qs = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vorticity = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
depth_buffer = ti.field(dtype=ti.f32, shape=res)
filtered_depth_buffer = ti.field(dtype=ti.f32, shape=res)
normal_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
thickness_buffer = ti.field(dtype=ti.f32, shape=res)
filtered_thickness_buffer = ti.field(dtype=ti.f32, shape=res)
light_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=res)
image = ti.Vector.field(3, dtype=ti.f32, shape=res)

# Wacky check table for grid
num_particle_in_grid = ti.field(dtype=ti.i32, shape=(grid_rows, grid_cols, grid_layers))
table_grid = ti.field(dtype=ti.i32, shape=(grid_rows, grid_cols, grid_layers, max_particle_in_grid))

# Wacky check table for neighbour
num_nb = ti.field(dtype=ti.i32, shape=num_particles)
table_nb = ti.field(dtype=ti.i32, shape=(num_particles, max_neighbour))

# LNM
visualize_lnm = True
grid_nb_cnt = ti.field(dtype=ti.i32, shape=num_grids)
grid_l = ti.field(dtype=ti.i32, shape=num_grids)
grid_n = ti.field(dtype=ti.i32, shape=num_grids)
grid_c = ti.Vector.field(3, dtype=ti.f32, shape=num_grids) # grid's centorid
grid_nb = ti.field(dtype=ti.i32, shape=(num_grids, 27))
particle_grid = ti.field(dtype=ti.i32, shape=num_particles)
# --------------------FUNCTIONS--------------------


@ti.func
def poly6(dist):
    # dist is a VECTOR
    result = 0.0
    d = dist.norm()
    if 0 < d < h:
        rhs = (h_2 - d * d) * (h_2 - d * d) * (h_2 - d * d)
        result = poly6_Coe * rhs / h_9
    return result


@ti.func
def poly6_scalar(dist):
    # dist is a SCALAR
    result = 0.0
    d = dist
    if 0 < d < h:
        rhs = (h_2 - d * d) * (h_2 - d * d) * (h_2 - d * d)
        result = poly6_Coe * rhs / h_9
    return result


@ti.func
def spiky(dist):
    # dist is a VECTOR
    result = ti.Vector([0.0, 0.0, 0.0])
    d = dist.norm()
    if 0 < d < h:
        m = (h - d) * (h - d)
        result = (spiky_Coe * m / (h_6 * d)) * dist
    return result
    # -Switch to 3D Vector when running in 3D
    # return ti.Vector([0.0, 0.0, 0.0])


@ti.func
def S_Corr(dist):
    upper = poly6(dist)
    lower = poly6_scalar(S_Corr_delta_q)
    m = upper/lower
    return -1.0 * S_Corr_k * m * m * m * m


@ti.func
def boundary_condition(v):
    # position filter
    # v is the position in vector form
    lower = radius
    upper = world[0] - radius
    # ---True Boundary---
    if v[0] <= lower:
        v[0] = lower + ti.random() * b_epsilon
    elif upper <= v[0]:
        v[0] = upper - ti.random() * b_epsilon
    if v[1] <= lower:
        v[1] = lower + ti.random() * b_epsilon
    elif upper <= v[1]:
        v[1] = upper - ti.random() * b_epsilon
    if v[2] <= lower:
        v[2] = lower + ti.random() * b_epsilon
    elif upper <= v[2]:
        v[2] = upper - ti.random() * b_epsilon
    return v


@ti.func
def get_grid(cord):
    new_cord = boundary_condition(cord)
    g_x = int(new_cord[0] / grid_size)
    g_y = int(new_cord[1] / grid_size)
    g_z = int(new_cord[2] / grid_size)
    return g_x, g_y, g_z

# --------------------KERNELS--------------------
# avoid nested for loop of position O(n^2)!


@ti.kernel
def pbf_prep():
    # ---save position---
    for p in position:
        last_position[p] = position[p]


@ti.kernel
def pbf_apply_force(ad: float, ws: float):
    # ---apply gravity/forces. Update velocity---
    gravity = ti.Vector([0.0, 0.0, -9.8])
    ad_force = ti.Vector([5.0, 0.0, 0.0])
    ws_force = ti.Vector([0.0, 5.0, 0.0])
    for i in velocity:
        velocity[i] += dt * (gravity + ad * ad_force + ws * ws_force + vorticity[i])
        # ---predict position---
        position[i] += dt * velocity[i]


@ti.kernel
def pbf_neighbour_search():
    # ---clean tables---
    for I in ti.grouped(num_particle_in_grid):
        num_particle_in_grid[I] = 0
    for I in ti.grouped(table_grid):
        table_grid[I] = -1
    for i in num_nb:
        num_nb[i] = 0
    for i, j in table_nb:
        table_nb[i, j] = -1
    for i in vorticity:
        vorticity[i] = ti.Vector([0.0, 0.0, 0.0])
    # ---update grid---
    for p in position:
        pos = position[p]
        p_grid = get_grid(pos)

        g_index = ti.atomic_add(num_particle_in_grid[p_grid[0], p_grid[1], p_grid[2]], 1)
        # ---ERROR CHECK---
        if g_index >= max_particle_in_grid:
            print("Grid overflows.")
        table_grid[p_grid[0], p_grid[1], p_grid[2], g_index] = p
    # ---update neighbour---
    for p in position:
        pos = position[p]
        p_grid = get_grid(pos)
        # nb_grid = neighbour_gird(p_grid)
        for off_x in ti.static(range(-1, 2)):
            for off_y in ti.static(range(-1, 2)):
                for off_z in ti.static(range(-1, 2)):
                    if 0 <= p_grid[0] + off_x < grid_cols:
                        if 0 <= p_grid[1] + off_y < grid_rows:
                            if 0 <= p_grid[2] + off_z < grid_layers:
                                nb_f32 = (p_grid[0] + off_x, p_grid[1] + off_y, p_grid[2] + off_z)
                                # translate nb as the i32 version of nb_f32
                                nb = [0, 0, 0]
                                nb[0] = ti.cast(nb_f32[0], ti.i32)
                                nb[1] = ti.cast(nb_f32[1], ti.i32)
                                nb[2] = ti.cast(nb_f32[2], ti.i32)
                                # print(nb, num_particle_in_grid.shape)
                                for i in range(num_particle_in_grid[nb[0], nb[1], nb[2]]):
                                    new_nb = table_grid[nb[0], nb[1], nb[2], i]
                                    n_index = ti.atomic_add(num_nb[p], 1)
                                    # ---ERROR CHECK---
                                    if n_index >= max_neighbour:
                                        print("Neighbour overflows.")
                                    table_nb[p, n_index] = new_nb


@ti.kernel
def pbf_solve():
    # ---Calculate lambdas---
    for p in position:
        pos = position[p]
        lower_sum = 0.0
        p_i = 0.0
        spiky_i = ti.Vector([0.0, 0.0, 0.0])
        for i in range(num_nb[p]):
            # ---Poly6---
            nb_index = table_nb[p, i]
            nb_pos = position[nb_index]
            p_i += mass * poly6(pos - nb_pos)
            # ---Spiky---
            s = spiky(pos - nb_pos) / rest_density
            spiky_i += s
            lower_sum += s.dot(s)
        constraint = (p_i / rest_density) - 1.0
        lower_sum += spiky_i.dot(spiky_i)
        lambdas[p] = -1.0 * (constraint / (lower_sum + lambda_epsilon))
    # ---Calculate delta Q---
    for p in position:
        delta_q = ti.Vector([0.0, 0.0, 0.0])
        pos = position[p]
        for i in range(num_nb[p]):
            nb_index = table_nb[p, i]
            nb_pos = position[nb_index]
            # ---S_Corr---
            scorr = S_Corr(pos - nb_pos)
            left = lambdas[p] + lambdas[nb_index] + scorr
            right = spiky(pos - nb_pos)
            delta_q += left * right / rest_density
        delta_qs[p] = delta_q
    # ---Update position with delta Q---
    for p in position:
        position[p] += delta_qs[p]


@ti.kernel
def pbf_update():
    # ---Update Position---
    for p in position:
        position[p] = boundary_condition(position[p])
    # ---Update Velocity---
    for v in velocity:
        velocity[v] = (position[v] - last_position[v]) / dt
    # ---Confinement/ XSPH Viscosity---
    # ---Using wacky gradient approximation for omega---
    for p in position:
        pos = position[p]
        xsph_sum = ti.Vector([0.0, 0.0, 0.0])
        omega_sum = ti.Vector([0.0, 0.0, 0.0])
        # -For Gradient Approx.-
        dx_sum = ti.Vector([0.0, 0.0, 0.0])
        dy_sum = ti.Vector([0.0, 0.0, 0.0])
        dz_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dx_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dy_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dz_sum = ti.Vector([0.0, 0.0, 0.0])
        dx = ti.Vector([g_del, 0.0, 0.0])
        dy = ti.Vector([0.0, g_del, 0.0])
        dz = ti.Vector([0.0, 0.0, g_del])
        for i in range(num_nb[p]):
            nb_index = table_nb[p, i]
            nb_pos = position[nb_index]
            v_ij = velocity[nb_index] - velocity[p]
            dist = pos - nb_pos
            # ---Vorticity---
            omega_sum += v_ij.cross(spiky(dist))
            # -Gradient Approx.-
            dx_sum += v_ij.cross(spiky(dist + dx))
            dy_sum += v_ij.cross(spiky(dist + dy))
            dz_sum += v_ij.cross(spiky(dist + dz))
            n_dx_sum += v_ij.cross(spiky(dist - dx))
            n_dy_sum += v_ij.cross(spiky(dist - dy))
            n_dz_sum += v_ij.cross(spiky(dist - dz))
            # ---Viscosity---
            poly = poly6(dist)
            xsph_sum += poly * v_ij
        # ---Vorticity---
        n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * g_del)
        n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * g_del)
        n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * g_del)
        n = ti.Vector([n_x, n_y, n_z])
        big_n = n.normalized()
        if not omega_sum.norm() == 0.0:
            vorticity[p] = vorti_epsilon * big_n.cross(omega_sum)
        # ---Viscosity---
        xsph_sum *= xsph_c
        velocity[p] += xsph_sum


@ti.func
def calculate_perspective_position(v):
    tan_x = 0.0
    tan_y = 0.0
    if TYPE == 1:
        v1 = v - (10.0, 10.0, 50.0)
        tan_x = ti.math.atan2(v1[0], -v1[2]) / fov * 2
        tan_y = ti.math.atan2(v1[1], -v1[2]) / fov * 2
    elif TYPE == 2:
        v1 = v - (10.0, -30.0, 10.0)
        tan_x = ti.math.atan2(v1[0], v1[1]) / fov * 2
        tan_y = ti.math.atan2(v1[2], v1[1]) / fov * 2
    return ti.Vector([tan_x + 0.5, tan_y + 0.5])


@ti.func
def project_to_screen(v):
    return ti.Vector([int(v[0] * res[0]), int(v[1] * res[1])])


@ti.func
def calculate_distance(v):
    camera = ti.Vector([0.0, 0.0, 0.0])
    if TYPE == 1:
        camera = ti.Vector([10.0, 10.0, 50.0])
    elif TYPE == 2:
        camera = ti.Vector([10.0, -30.0, 10.0])
    return (v - camera).norm()


@ti.func
def perspective_radius(r, d):
    return r / d


@ti.func
def add_point_to_depth_buffer(screen_pos, r, distance):
    r = int(r)
    upper_x = ti.math.max(screen_pos[0] + r, res[0] - 1)
    lower_x = ti.math.max(screen_pos[0] - r, 0)
    upper_y = ti.math.max(screen_pos[1] + r, res[1] - 1)
    lower_y = ti.math.max(screen_pos[1] - r, 0)
    for i in range(lower_x, upper_x):
        for j in range(lower_y, upper_y):
            r1 = (i - screen_pos[0]) * (i - screen_pos[0]) + (j - screen_pos[1]) * (j - screen_pos[1])
            if r1 < r * r:
                thickness: ti.f32 = 2.0 * ti.math.sqrt(r * r - r1) / r * visual_radius
                depth = distance - thickness
                if unsafe:
                    depth_buffer[i, j] = ti.min(depth_buffer[i, j], depth)
                else:
                    ti.atomic_min(depth_buffer[i, j], depth)


@ti.func
def add_point_to_thickness_buffer(screen_pos, r):
    r = int(r)
    upper_x = ti.math.max(screen_pos[0] + r, res[0] - 1)
    lower_x = ti.math.max(screen_pos[0] - r, 0)
    upper_y = ti.math.max(screen_pos[1] + r, res[1] - 1)
    lower_y = ti.math.max(screen_pos[1] - r, 0)
    for i in range(lower_x, upper_x):
        for j in range(lower_y, upper_y):
            r1 = (i - screen_pos[0]) * (i - screen_pos[0]) + (j - screen_pos[1]) * (j - screen_pos[1])
            if r1 < r * r:
                thickness: ti.f32 = 2.0 * ti.math.sqrt(r * r - r1) / r * visual_radius
                ti.atomic_add(thickness_buffer[i, j], thickness)


@ti.func
def depth_for_display(depth):
    return (depth - 20.0) / 80.0


@ti.func
def thickness_for_display(thickness):
    return ti.exp(-thickness / 20.0)


DISTANCE_COEFFICIENT = 2.0
VALUE_FALLOFF_COEFFICIENT = 500.0


@ti.func
def calculate_filter_buffer(buffer, i, j):
    # use bilateral filter
    # https://en.wikipedia.org/wiki/Bilateral_filter
    upper_x = ti.math.min(i + filter_radius, res[0] - 1)
    lower_x = ti.math.max(i - filter_radius, 0)
    upper_y = ti.math.min(j + filter_radius, res[1] - 1)
    lower_y = ti.math.max(j - filter_radius, 0)
    sum_weight = 0.0
    sum_depth = 0.0
    for x in range(lower_x, upper_x):
        for y in range(lower_y, upper_y):
            distance_weight = ti.exp(-((x - i) * (x - i) + (y - j) * (y - j)) /
                                     (DISTANCE_COEFFICIENT * filter_radius * filter_radius))
            value_weight = ti.exp(-((buffer[x, y] - buffer[i, j]) *
                                    (buffer[x, y] - buffer[i, j])) /
                                  VALUE_FALLOFF_COEFFICIENT)
            weight = distance_weight * value_weight
            sum_weight += weight
            sum_depth += weight * buffer[x, y]
    return sum_depth / sum_weight


@ti.func
def calculate_normal_buffer(i, j):
    # calculate normal buffer
    upper_x = ti.math.min(i + 1, res[0] - 1)
    lower_x = ti.math.max(i - 1, 0)
    upper_y = ti.math.min(j + 1, res[1] - 1)
    lower_y = ti.math.max(j - 1, 0)
    dx: ti.f32 = (filtered_depth_buffer[upper_x, j] - filtered_depth_buffer[lower_x, j]) / (upper_x - lower_x)
    dy: ti.f32 = (filtered_depth_buffer[i, upper_y] - filtered_depth_buffer[i, lower_y]) / (upper_y - lower_y)
    length_factor = 1.0 / res[0] * filtered_depth_buffer[i, j]
    normal = ti.Vector([-dx, -1.0 * length_factor, -dy])
    return normal.normalized()


@ti.func
def calculate_attenuation(i, j):
    color = ti.Vector([1.0, 1.0, 1.0])
    if filtered_thickness_buffer[i, j] > 0.01:
        color = ti.Vector([
            color[0] * ti.exp(-filtered_thickness_buffer[i, j] * fluid_filter_color[0] * fluid_attenuation_coefficient),
            color[1] * ti.exp(-filtered_thickness_buffer[i, j] * fluid_filter_color[1] * fluid_attenuation_coefficient),
            color[2] * ti.exp(-filtered_thickness_buffer[i, j] * fluid_filter_color[2] * fluid_attenuation_coefficient),
            ])
    return color


@ti.func
def mirror_vector(normal, vector):
    vector_n = vector.normalized()
    normal_n = normal.normalized()
    return vector_n - 2.0 * normal.dot(vector) * normal_n


@ti.func
def angle(v1, v2):
    return ti.acos(ti.abs(v1.normalized().dot(v2.normalized())))


@ti.func
def calculate_reflection(i, j):
    color = light_color
    if depth_buffer[i, j] < 100.0:
        reflect_vector = mirror_vector(normal_buffer[i, j], camera_angle)
        reflect_angle = angle(reflect_vector, camera_angle)
        if reflect_angle > light_angle:
            color = background_color
    return ti.Vector([
        ti.math.min(color[0], 1.0),
        ti.math.min(color[1], 1.0),
        ti.math.min(color[2], 1.0)])


@ti.func
def reflection_coefficient(theta):
    r0 = (1.5 - 1.0) / (1.5 + 1.0)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ((1.0 - ti.cos(theta)) ** 5)


@ti.func
def calculate_color(i, j):
    reflect = reflection_coefficient(angle(normal_buffer[i, j], camera_angle))
    return calculate_reflection(i, j) * reflect + light_attenuation[i, j] * (1 - reflect)


@ti.func
def add_particle_to_buffer(i):
    origin_pos = position[i]
    screen_pos = project_to_screen(calculate_perspective_position(origin_pos))
    distance = calculate_distance(origin_pos)
    screen_radius = perspective_radius(visual_radius, distance) * res[0]
    add_point_to_depth_buffer(screen_pos, screen_radius, distance)
    add_point_to_thickness_buffer(screen_pos, screen_radius)


@ti.kernel
def generate_render_buffer():
    # init buffers
    for i, j in depth_buffer:
        depth_buffer[i, j] = 100.0
    for i, j in thickness_buffer:
        thickness_buffer[i, j] = 0.0
    # make the l1_position
    new_index = 0
    if visualize_lnm:
        for i in position:
            if grid_l[particle_grid[i]] == 1:
                index = ti.atomic_add(new_index, 1)
                l1_points[index] = i
    if visualize_lnm:
        for i in range(new_index):
            add_particle_to_buffer(l1_points[i])
    else:
        for i in position:
            add_particle_to_buffer(i)
    for i, j in filtered_depth_buffer:
        filtered_depth_buffer[i, j] = calculate_filter_buffer(depth_buffer, i, j)
    for i, j in normal_buffer:
        normal_buffer[i, j] = calculate_normal_buffer(i, j)
    for i, j in filtered_thickness_buffer:
        filtered_thickness_buffer[i, j] = calculate_filter_buffer(thickness_buffer, i, j)
    for i, j in light_attenuation:
        light_attenuation[i, j] = calculate_attenuation(i, j)
    for i, j in image:
        image[i, j] = calculate_color(i, j)


def pbf(ad, ws):
    pbf_prep()
    pbf_apply_force(ad, ws)
    pbf_neighbour_search()
    for _ in range(solve_iteration):
        pbf_solve()
    
    pbf_update()
    # LNM algorithm 
    lnm()
    generate_render_buffer()


@ti.kernel
def init():
    for i in position:
        pos_x = 2 + 0.8 * (i % 20)
        pos_y = 2 + 0.8 * ((i % 400) // 20)
        pos_z = 1 + 0.8 * (i // 400)
        position[i] = ti.Vector([pos_x, pos_y, pos_z])
        vorticity[i] = ti.Vector([0.0, 0.0, 0.0])


# def render(gui: ti.GUI):
#     gui.clear(background_color)
#     render_position = position.to_numpy()
#     render_position /= boundary
#     gui.circles(render_position, radius=visual_radius, color=particle_color)
#     gui.show()


def parameter_init():
    global light_position
    if TYPE == 1:
        light_position = ti.Vector([0.0, -1.0, 0.0])
    elif TYPE == 2:
        light_position = ti.Vector([0.0, 0.0, -1.0])


@ti.func
def get_lnm_grid(cord):
    new_cord = boundary_condition(cord)
    g_x = int(new_cord[0] / lnm_grid_size)
    g_y = int(new_cord[1] / lnm_grid_size)
    g_z = int(new_cord[2] / lnm_grid_size)
    return g_x, g_y, g_z

@ti.func
def get_lnm_grid_idx(i, j, k):
    return (i * lnm_grid_rows + j) * lnm_grid_cols + k


@ti.kernel
def lnm_init():
    for i in range(lnm_grid_rows):
        for j in range(lnm_grid_cols):
            for k in range(lnm_grid_layers):
                idx = get_lnm_grid_idx(i, j, k)
                cnt = 0
                for off_x in ti.static(range(-1, 2)):
                    for off_y in ti.static(range(-1, 2)):
                        for off_z in ti.static(range(-1, 2)):
                            if 0 <= i + off_x < lnm_grid_cols:
                                if 0 <= j + off_y < lnm_grid_rows:
                                    if 0 <= k + off_z < lnm_grid_layers:
                                        grid_nb[idx, cnt] = get_lnm_grid_idx(i + off_x, j + off_y, k + off_z)
                                        cnt += 1
                grid_nb_cnt[idx] = cnt


@ti.kernel
def lnm():
    # https://github.com/felpzOliveira/Bubbles/blob/76f72b36bfdd3eabc9c43be62fba36997b197629/src/boundaries/lnm.h#L333
    grid_l.fill(0)
    grid_n.fill(0)
    grid_c.fill(0)
    
    for p in position:
        pos = position[p]
        p_grid = get_lnm_grid(pos)
        p_grid_idx = get_lnm_grid_idx(p_grid[0], p_grid[1], p_grid[2])
        particle_grid[p] = p_grid_idx
        grid_n[p_grid_idx] += 1
        grid_c[p_grid_idx] += pos
        
    for idx in grid_n:
        if grid_n[idx] == 0:
            continue
        grid_c[idx] /= grid_n[idx]
        
    dim = 3
    threshold = pow(3, dim)
    particle_threshold = pow(2, dim)
    for idx in grid_n:
        if grid_n[idx] == 0:
            continue
        if grid_nb_cnt[idx] != threshold:
            grid_l[idx] = 1
        else:
            for l in range(grid_nb_cnt[idx]):
                nb_idx = grid_nb[idx, l]
                if nb_idx != idx and grid_n[nb_idx] == 0:
                    grid_l[idx] = 1
                    break
    for idx in grid_n:
        if grid_n[idx] == 0 or grid_l[idx] == 1:
            continue
        for l in range(grid_nb_cnt[idx]):
            nb_idx = grid_nb[idx, l]
            if nb_idx != idx and grid_l[nb_idx] == 1 and grid_n[nb_idx] < particle_threshold:
                grid_l[idx] = 2
                break


def main():
    parameter_init()
    lnm_init()
    init()
    # prefix = "./3d_ply/a.ply"
    # if not os.path.exists(os.path.dirname(prefix)):
    #     os.makedirs(os.path.dirname(prefix))
    gui = ti.GUI('PBF3D', res)
    frame_count = 0
    while gui.running:
        # ---Control Waves---
        ad = 0.0
        ws = 0.0
        gui.get_event()
        if gui.is_pressed('a'):
            ad = -1.0
        elif gui.is_pressed('d'):
            ad = 1.0
        if gui.is_pressed('w'):
            ws = -1.0
        elif gui.is_pressed('s'):
            ws = 1.0
        pbf(ad, ws)
        # ---Record 3D result---
        # if frame_count > -1:
        #     np_pos = np.reshape(position.to_numpy(), (num_particles, 3))
        #     writer = ti.tools.PLYWriter(num_vertices=num_particles)
        #     writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        #     writer.export_frame(frame_count, prefix)
        gui.set_image(image)
        gui.show()
        # ---Frame Control---
        if frame_count % 100 == 0:
            print("Frame:".format(frame_count))
        frame_count += 1
    return 0


if __name__ == '__main__':
    main()
