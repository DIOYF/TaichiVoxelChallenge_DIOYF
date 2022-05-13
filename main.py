from scene import Scene
import taichi as ti
from taichi.math import *


scene = Scene(voxel_edges=0, exposure=1)  ##  设置边界和曝光度
scene.set_floor(-60, (0.5, 1.0, 1.0))  ##  设置地板高度，设置地板颜色
scene.set_background_color((0.52, 0.87, 0.92))  ##  设置背景颜色
scene.set_directional_light((1, 1, 1), 0.2, (0.8, 0.6, 0.3))  ## 设置光照
pi = 3.1415926


@ti.func ## rotate复用，rotate包含不变量x-1,y-2,z-3,不转-0和旋转角度theta两个参数,theta采用弧度制
def set_rotate(rotate, vector):
    sin_theta = ti.sin(float(rotate[1]))
    cos_theta = ti.cos(float(rotate[1]))
    vec = ti.Vector((1.0, 1.0, 1.0))
    if rotate[0] == 1:
        vec = ti.Matrix([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]], ti.f32) @ vector
    elif rotate[0] == 2:
        vec = ti.Matrix([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]], ti.f32) @ vector
    elif rotate[0] == 3:
        vec = ti.Matrix([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]], ti.f32) @ vector
    else:
        vec = vector
    return vec


@ti.func ## 颜色加噪单独写出来
def noisy_color(color, color_noise_val):
    return color + vec3(color_noise_val) * ti.random()


@ti.func ## 小矩形框架
def set_rectangle(center, scale, color, rotate):
    for i, j, k in ti.ndrange((-scale, scale), (-scale, scale), (-1, 1)):
        scene.set_voxel(center + set_rotate(rotate, vec3(i, j, k)), 1, color)
    for _i, _j, _k in ti.ndrange((-scale+1, scale - 1), (-scale+1, scale - 1), (-1, 1)):
        scene.set_voxel(center + set_rotate(rotate, vec3(_i, _j, _k)), 0, color)


@ti.func ## 底座曲面模块
def set_curve_surface(center, scale, weight, color, color_noise, rotate):
    for i, j in ti.ndrange((0, scale+1), (0, scale+1)):
        k = scale/2 * ti.sin(weight * (float(j) / 60) * float(i))
        if (vec2(i, j)-vec2(scale, scale)).dot(vec2(i, j)-vec2(scale, scale)) < scale*scale:
            scene.set_voxel(center + set_rotate(rotate, vec3(i, k, j)), 1, noisy_color(color, color_noise))


@ti.func ## 风车风扇
def set_fan(center, scale):
    color = vec3(0.89, 0.87, 0.88)
    for i, j, k in ti.ndrange((-scale / 2, scale / 2), (-scale / 2, scale / 2), (-1, 1)):
        x = vec2(i, j)
        if x.dot(x) < scale * scale / 4:
            scene.set_voxel(center + set_rotate((0, 0), vec3(i, j, k)), 1, noisy_color(color, 0.2))
    for _i, _j, _k in ti.ndrange((-scale / 8, scale / 8), (0, scale * 2), (-2, -1)):
        y = vec3(_i, _j, _k)
        scene.set_voxel(center + set_rotate((3, pi / 4), y), 1, noisy_color(color, 0.2))
        scene.set_voxel(center + set_rotate((3, pi * 3 / 4), y), 1, noisy_color(color, 0.2))
        scene.set_voxel(center + set_rotate((3, pi * 5 / 4), y), 1, noisy_color(color, 0.2))
        scene.set_voxel(center + set_rotate((3, pi * 7 / 4), y), 1, noisy_color(color, 0.2))
    for __i, __j, in ti.ndrange((-1, 1), (3, 13)):
        pos = vec3(__i * 4, __j * 4, 0)
        set_rectangle(center + set_rotate((3, pi / 4), pos), 2, vec3(1.0, 0.72, 0.8), (3, pi/4))
        set_rectangle(center + set_rotate((3, pi * 3 / 4), pos), 2, vec3(0.8, 0.93, 0.9), (3, pi * 3/4))
        set_rectangle(center + set_rotate((3, pi * 5 / 4), pos), 2, vec3(1.0, 0.95, 0.88), (3, pi * 5/4))
        set_rectangle(center + set_rotate((3, pi * 7 / 4), pos), 2, vec3(0.52, 0.8, 0.84), (3, pi * 7/4))


@ti.func ## 风车的基座
def set_house(center, scale):
    for i in range(-50, 20):
        for j, k in ti.ndrange((-scale+i//10, scale-i//10), (-scale+i//10, scale-i//10)):
            scene.set_voxel(center + vec3(j, i, k), 1, noisy_color(vec3(0.8, 0.6, 0.1), 0.2))
            scene.set_voxel(center + vec3(j, i, k), 1, noisy_color(vec3(1.0, 0.0, 0.0), 0.2))
        for m, n in ti.ndrange((-2, 2), (4, 12)):
            scene.set_voxel(center + vec3(m, 15, n), 1, noisy_color(vec3(0.89, 0.87, 0.88), 0.2))


@ti.kernel
def initialize_voxels():
    # 参数：中心坐标，尺度，颜色，颜色噪声，旋转情况
    set_curve_surface(ivec3(-50, -60, -50), 50, 0.05, vec3(0.5, 1.0, 0), 0.25, (0, 0))
    set_curve_surface(ivec3(-50, -60, 50), 50, 0.05, vec3(0.5, 1.0, 0), 0.25, (2, pi/2))
    set_curve_surface(ivec3(50, -60, 50), 50, 0.05, vec3(0.5, 1.0, 0), 0.25, (2, pi))
    set_curve_surface(ivec3(50, -60, -50), 50, 0.05, vec3(0.5, 1.0, 0), 0.25, (2, pi*3/2))
    set_fan(ivec3(0, 15, 12), 10)   # 参数：中心坐标，尺度
    set_house(ivec3(0, 0, 0), 5)   # 参数：中心坐标，尺度


initialize_voxels()
scene.finish()

