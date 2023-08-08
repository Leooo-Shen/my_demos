import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def rotate_along_axis(Q, angle=5, axis='x'):
    # input Q should have shape (4,)
    rot_matrix_origin = R.from_quat(Q).as_matrix()
    m_pos = R.from_euler(axis, angle, degrees=True).as_matrix()
    m_neg = R.from_euler(axis, -angle, degrees=True).as_matrix()

    rot_matrix_tool_pos = np.dot(m_pos, rot_matrix_origin) 
    rot_matrix_tool_neg = np.dot(m_neg, rot_matrix_origin)
    
    Q_pos = R.from_matrix(rot_matrix_tool_pos).as_quat()
    Q_neg = R.from_matrix(rot_matrix_tool_neg).as_quat()
    movements = (Q_pos, Q_neg, Q)
    return movements


def fake_trajectory(type):
    """
    Get the predifined trajectory that consists a few target points

    Returns:
        list: a list of tuples that defines the trajectory
    """
    x = 0.500
    y = 0.0
    z = 0.300 # height
    step = 0.1
    
    if type == 'square':
        trajectory = [(x, y, z), (x-step, y, z), (x-step, y+step, z), (x+step, y+step, z), (x+step, y, z), (x, y, z)]
    elif type =='s':
        trajectory = [(x, y, z), (x, y-step, z), (x-step, y-step, z)]

        
    return trajectory


def reach_target_position(current, target, epsilon):
    if (abs(current.x - target.x) < epsilon) and (abs(current.y - target.y) < epsilon) and (abs(current.z - target.z) < epsilon):
        return True


def reach_target_orientation(current, target, epsilon):
    if (abs(current.x - target.x) < epsilon) and (abs(current.y - target.y) < epsilon) and (abs(current.z - target.z) < epsilon) and (current.w - target.w < epsilon):
        return True


def interpolate_num(trajectory, num_points=10):
    """interpolate the predefined trajectory with more points

    Args:
        trajectory (list): list of tuples representing predefined trajectory [(x0, y0, z0)]. All trajectory points in mm!!
        stepsize (float): fixed stepsize for interpolation

    Returns:
        res (list): interpolated trajectort
    """
    res = []
    
    for idx in range(len(trajectory) - 1):
        # avoid two repetition points
        if trajectory[idx] != trajectory[idx+1]:
            x0, y0, z0 = trajectory[idx][0], trajectory[idx][1], trajectory[idx][2]
            x1, y1, z1 = trajectory[idx+1][0], trajectory[idx+1][1], trajectory[idx+1][2]
            x_intep = np.linspace(x0, x1, num_points) if x0 != x1 else [x0] * num_points
            y_intep = np.linspace(y0, y1, num_points) if y0 != y1 else [y0] * num_points
            z_intep = np.linspace(z0, z1, num_points) 
            res += list(zip(x_intep, y_intep, z_intep))
            print(res)
        else:
            res += trajectory[idx]
    return res
    
    
def interpolate_stepsize(points, step_size):
    interpolated_points = []
    for i in range(len(points) - 1):
        start_point = np.array(points[i])
        end_point = np.array(points[i + 1])
        dist = np.linalg.norm(end_point - start_point)
        num_steps = int(dist / step_size)
        for j in range(num_steps + 1):
            t = j / num_steps
            interpolated_point = start_point * (1 - t) + end_point * t
            interpolated_points.append(interpolated_point.tolist())
    return interpolated_points
    

def vis_trajectory(t_origin, t_intep, projection, diagonal_points=None, savefig=False):
    if projection == '2d':
        x = np.array([point[0] for point in t_intep])
        y = np.array([point[1] for point in t_intep])
        s = [20] * len(x) # size of the points
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(x, y, s, c='orange', alpha=0.5, label='interpolated')
        ax.plot(x, y)
        
        if t_origin is not None and len(t_origin) != 0:
            x_origin = np.array([point[0] for point in t_origin])
            y_origin = np.array([point[1] for point in t_origin])
            s_origin = [20] * len(x_origin) # size of the points
            ax.scatter(x_origin, y_origin, s_origin, c='red', alpha=1, label='region centers')
        if diagonal_points is not None:
            x = np.array([point[0] for point in diagonal_points])
            y = np.array([point[1] for point in diagonal_points])
            s = [20] * len(diagonal_points) # size of the points
            ax.scatter(x, y, s, c='black', alpha=1, label='detected diagonal points')
            print((max(x) - min(x))/3)
            plt.xticks(np.arange(min(x), max(x), (max(x) - min(x))/3))
            plt.yticks(np.arange(min(y), max(y), (max(y) - min(y))/3))
        
    elif projection == '3d':
        x = np.array([point[0] for point in t_intep])
        y = np.array([point[1] for point in t_intep])
        z = np.array([point[2] for point in t_intep])
        s = [200] * len(x) # size of the points
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, s, c='orange', alpha=0.5, label='interpolated')
        plt.plot(x,y,z)
        
        # visualize the original points
        if t_origin is not None and len(t_origin) != 0:
            x_origin = np.array([point[0] for point in t_origin])
            y_origin = np.array([point[1] for point in t_origin])
            z_origin = np.array([point[2] for point in t_origin])
            s_origin = [200] * len(x_origin) # size of the points
            ax.scatter(x_origin, y_origin, z_origin, s_origin, c='red', alpha=1, label='region centers')
    else:
        print('projection should be either 2d or 3d')
    plt.grid()
    plt.legend(loc='upper right', fontsize=10)
    if savefig:
        plt.savefig('trajectory.png')
        print(' > trajectory saved to trajectory.png')
    else:
        plt.show()


def create_grid(p1, p2, num_cells=(3,3), shape='s'):
    """
    Creates a grid of points given the diagonal corners of the grid.
    :param p1(tuple): Diagonal corner 1, can be either 2D or 3D
    :param p2(tuple): Diagonal corner 2, can be either 2D or 3D
    :param cell_size(int): size of the grid cells
    :param shape(string): shape of the grid, either 's' or 'parallel'
        for s: the grid will be connected in an s shape
        for parallel: the grid will be connected in a parallel shape
    return: list of points
        if p1 and p2 are 2D, the grid will be 2D
        elif p1 and p2 are 3D, the grid will be 3D, with z as the average of p1.z and p2.z
    
    """
    
    # calculate the cell size in each direction
    num_cells_x = num_cells[0]
    num_cells_y = num_cells[1]
    
    cell_size_x =abs(p1[0] - p2[0]) / num_cells[0]
    cell_size_y =abs(p1[1] - p2[1]) / num_cells[1]
    
    # calculate the starting point for the grid
    start_x = min(p1[0], p2[0])
    start_y = min(p1[1], p2[1])
    
    # create a list to store the center points of each cell
    cell_centers = []
    
    if shape == 's':
        # connect the grid center in s shape
        for i in range(num_cells_x):
            row = []
            for j in range(num_cells_y):
                center_x = start_x + (i + 0.5) * cell_size_x
                center_y = start_y + (j + 0.5) * cell_size_y
                row.append((center_x, center_y))
            if i % 2 == 0:
                cell_centers += row
            else:
                cell_centers += row[::-1]
                
    elif shape == 'parallel':
        # connect the grid center in parallel shape
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                center_x = start_x + (i + 0.5) * cell_size_x
                center_y = start_y + (j + 0.5) * cell_size_y
                cell_centers.append((center_x, center_y))
    # if p1, p2 are 3D point, use the average of z value as the z value of the all center points
    if len(p1) == 3 and len(p2) == 3:
        avg_z = (p1[2] + p2[2]) / 2
        cell_centers = [(point[0], point[1], avg_z) for point in cell_centers]
    return cell_centers


def visualize_log(log_path):
    df = pd.read_csv(log_path)
    print(df.head())
    # create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=['r' if s == 1 else 'b' for s in df['sweep']])

    # set axis labels and plot title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Scatter Plot')
    plt.show()

    
    
# if __name__ == "__main__":
    
    # p1 = np.array([412, -150, 350]) * 1e-3
    # p2 = np.array([620, 170, 350]) * 1e-3
    
    # centers = create_grid(p1, p2, num_cells=(3,3), shape='s')
    # print(centers)
    # traj = interpolate_stepsize(centers,1e-2)
    # vis_trajectory(centers, traj, projection='2d', diagonal_points=[p1, p2], savefig=True)

    # visualize_log('/home/demo/chengzhi/robotic-us/history_12:50:08.csv')