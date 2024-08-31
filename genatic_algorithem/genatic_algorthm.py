import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义心形函数，向右平移50个单位
def heart_function_shifted(t):
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return x + 50, y

# 定义拟合函数，使用正弦波
def sine_wave(t, params):
    a, b, c, d, e, f = params
    return a * np.sin(b * t + c) + d * np.sin(e * t + f)
# 定义适应度函数
def fitness(params):
    t = np.linspace(0, 2 * np.pi, 100)
    target_x, target_y = heart_function_shifted(t)
    fitted_y = sine_wave(t, params)
    loss = np.mean((fitted_y - target_y)**2)
    return loss

# 遗传算法参数
population_size = 100
params_per_individual = 6
num_generations = 10000
mutation_rate = 0.4
crossover_rate = 0.8
mutation_step_size = 0.01

# 初始化种群
population = np.random.rand(population_size, params_per_individual)
losses_over_time = []

# 设置绘图
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].set_xlim(0, 110)
ax[0].set_ylim(-30, 30)
ax[1].set_xlim(0, num_generations)
ax[1].set_ylim(0, 100)
line1, = ax[0].plot([], [], label='Heart Curve')
line2, = ax[0].plot([], [], label='Fitted Curve')
line3, = ax[1].plot([], [], label='Loss Over Time')

# 初始化文本对象
generation_text = ax[1].text(0.5, 0.9, '', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    generation_text.set_text('')
    return line1, line2, line3, generation_text

def select(population, losses):
    sorted_indices = np.argsort(losses)
    return population[sorted_indices]

def crossover(population, crossover_rate):
    offspring = []
    for i in range(0, population_size, 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, params_per_individual)
            offspring.append(np.concatenate((population[i][:crossover_point], population[i+1][crossover_point:])))
            offspring.append(np.concatenate((population[i+1][:crossover_point], population[i][crossover_point:])))
        else:
            offspring.append(population[i])
            offspring.append(population[i+1])
    return np.array(offspring)

def mutate(population, mutation_rate, mutation_step_size):
    mutation_indices = np.random.rand(*population.shape) < mutation_rate
    mutation_noise = np.random.normal(0, mutation_step_size, population.shape)
    population[mutation_indices] += mutation_noise[mutation_indices]
    return population

def update(frame):
    global population, losses_over_time

    # 计算适应度
    losses = np.apply_along_axis(fitness, 1, population)
    best_loss = np.min(losses)
    best_params = population[np.argmin(losses)]
    losses_over_time.append(best_loss)

    # 选择
    population = select(population, losses)

    # 交叉
    population = crossover(population, crossover_rate)

    # 变异
    population = mutate(population, mutation_rate, mutation_step_size)

    # 每10代刷新一次绘图
    if frame % 10 == 0:
        t = np.linspace(0, 2 * np.pi, 100)
        target_x, target_y = heart_function_shifted(t)
        fitted_curve = sine_wave(t, best_params)

        line1.set_data(target_x, target_y)
        line2.set_data(target_x, fitted_curve)
        line3.set_data(range(len(losses_over_time)), losses_over_time)
        print(f"Generation {frame}: Best Loss = {best_loss}, Params = {best_params}")
        # 更新文本对象的内容
        generation_text.set_text(f'Generation {frame}')

    return line1, line2, line3, generation_text

# 运行动画
ani = FuncAnimation(fig, update, frames=range(num_generations), init_func=init, blit=False)
plt.tight_layout()
plt.show()