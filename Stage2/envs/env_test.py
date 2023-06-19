from .env import Truss


if __name__ == '__main__':
    num_points = 6
    initial_state_files = 'best_results'
    coordinate_range = [(0.0, 18.288), (0.0, 9.144)]
    area_range = (6.452e-05, 0.02)
    coordinate_delta_range = [(-0.5715, 0.5715), (-0.5715, 0.5715)]
    area_delta_range = (-0.0005, 0.0005)
    fixed_points = 4
    variable_edges = -1
    max_refine_steps = 1000
    env = Truss(num_points, initial_state_files, coordinate_range, area_range, coordinate_delta_range, area_delta_range, fixed_points, variable_edges, max_refine_steps)
    while True:
        env.reset()
