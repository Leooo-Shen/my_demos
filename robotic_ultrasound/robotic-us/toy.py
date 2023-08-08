import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('toy.csv')

# create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], c=['r' if s == 1 else 'b' for s in df['sweep']])

# set axis labels and plot title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot')

# show plot
plt.show()
