import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from matplotlib.animation import FuncAnimation
import numpy as np


def draw_xyz_axes(axis_half_length: float = 5000.0) -> None:

	fig = plt.figure(figsize=(8, 8), dpi=120)
	ax = fig.add_subplot(111, projection='3d')
	# Black background for both figure and axes
	fig.patch.set_facecolor('black')
	ax.set_facecolor('black')

	# No axis lines; clean black background only

	# Interactive view state: zoom half-length and center
	view_half_len = axis_half_length
	center_x, center_y, center_z = 0.0, 0.0, 0.0
	initial_half_len = view_half_len
	initial_center = (center_x, center_y, center_z)
	# Allow much closer zoom-in (very small min window) and wide zoom-out
	min_half_len = 1e-3
	max_half_len = axis_half_length * 100.0

	# Initial limits based on current view window and center
	ax.set_xlim(center_x - view_half_len, center_x + view_half_len)
	ax.set_ylim(center_y - view_half_len, center_y + view_half_len)
	ax.set_zlim(center_z - view_half_len, center_z + view_half_len)

	# No axis arrows or labels

	# Keep a generous window but avoid a visible box; the line visually extends "beyond"
	# (Limits are now driven by interactive state above)

	# Equal aspect ratio for all axes
	ax.set_box_aspect((1, 1, 1))

	# Hide everything else (ticks, panes, spines, labels)
	for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
		axis.set_ticks([])
		axis.set_ticklabels([])

	# Remove background panes and grid
	ax.grid(False)
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False
	# Also remove pane edges for a clean, borderless look
	ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
	ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
	ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

	# Remove axis lines (spines) from viewbox; keep only our plotted lines
	for spine in ax.spines.values():
		spine.set_visible(False)

	# Turn off the outer 3D frame entirely (removes the cube-like black lines)
	ax.set_axis_off()

	# ----- Simulate and draw the system trajectory (thin white line) -----
	def f_xyz(state, a, b, c, d, e, f):
		x, y, z = state
		dx = (z - b) * x - d * y
		dy = d * x + (z - b) * y
		dz = c + a * z - (z ** 3) / 3.0 - (x ** 2 + y ** 2) * (1.0 + e * z) + f * z * (x ** 3)
		return np.array([dx, dy, dz], dtype=np.float64)

	def rk4_step(state, h, a, b, c, d, e, f):
		k1 = f_xyz(state, a, b, c, d, e, f)
		k2 = f_xyz(state + 0.5 * h * k1, a, b, c, d, e, f)
		k3 = f_xyz(state + 0.5 * h * k2, a, b, c, d, e, f)
		k4 = f_xyz(state + h * k3, a, b, c, d, e, f)
		return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

	# Default parameters (can be tuned)
	params = {
		'a': 0.95,
		'b': 0.7,
		'c': 0.6,
		'd': 3.5,
		'e': 0.25,
		'f': 0.1,
	}
	# Integration settings
	initial_state = np.array([0.1, 0.0, 0.0], dtype=np.float64)
	step_size = 0.01
	total_steps = 40000
	transient = 2000  # discard initial steps

	trajectory = np.empty((total_steps, 3), dtype=np.float64)
	state = initial_state.copy()
	for i in range(total_steps):
		state = rk4_step(state, step_size, **params)
		trajectory[i] = state

	# Prepare trajectory and compute center/bounds for initial view and brightness
	traj = trajectory[transient:]
	mins = np.min(traj, axis=0)
	maxs = np.max(traj, axis=0)
	center_x = float((mins[0] + maxs[0]) * 0.5)
	center_y = float((mins[1] + maxs[1]) * 0.5)
	center_z = float((mins[2] + maxs[2]) * 0.5)
	span = float(np.max(maxs - mins))
	initial_half_len = max(min_half_len, span * 0.55)
	initial_center = (center_x, center_y, center_z)

	# Increase point spacing by sub-sampling
	point_stride = 5  # larger => sparser points
	traj_ds = traj[::point_stride]
	# Base time colors (blue->yellow without reds) and distance-based brightness
	t_colors = np.linspace(0.0, 1.0, len(traj_ds), dtype=np.float64)
	cmap = plt.get_cmap('cividis')
	base_rgba = cmap(t_colors)
	dists = np.linalg.norm(traj_ds - np.array([center_x, center_y, center_z]), axis=1)
	dmin, dmax = float(np.min(dists)), float(np.max(dists))
	den = dmax - dmin if (dmax - dmin) > 1e-12 else 1.0
	brightness = 1.0 - (dists - dmin) / den
	brightness = np.clip(brightness, 0.7, 1.0)
	base_rgba[:, :3] = base_rgba[:, :3] * brightness[:, None]
	base_rgba[:, 3] = 1.0
	ax.scatter(traj_ds[:, 0], traj_ds[:, 1], traj_ds[:, 2], c=base_rgba, s=0.6,
			edgecolors='none')

	def update_view(new_half_len: float) -> None:
		nonlocal view_half_len, center_x, center_y, center_z
		view_half_len = float(np.clip(new_half_len, min_half_len, max_half_len))
		ax.set_xlim(center_x - view_half_len, center_x + view_half_len)
		ax.set_ylim(center_y - view_half_len, center_y + view_half_len)
		ax.set_zlim(center_z - view_half_len, center_z + view_half_len)
		fig.canvas.draw_idle()

	def on_scroll(event):
		if event.button == 'up':
			update_view(view_half_len * 0.8)
		elif event.button == 'down':
			update_view(view_half_len * 1.25)

	def on_key(event):
		key = (event.key or '').lower()
		if key in ['+', '=']:
			update_view(view_half_len * 0.8)
		elif key in ['-', '_']:
			update_view(view_half_len * 1.25)
		elif key in ['r']:
			center_x, center_y, center_z = initial_center
			update_view(initial_half_len)

	# Connect interactions
	cid_scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)
	cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

	# Apply the tight initial view to frame the trajectory
	update_view(initial_half_len)

	# Static formula label in the top-left (figure coords), white on black (three lines)
	line1 = r"$\frac{dx}{dt} = (z-b)\,x - d\,y$"
	line2 = r"$\frac{dy}{dt} = d\,x + (z-b)\,y$"
	line3 = r"$\frac{dz}{dt} = c + a\,z - \frac{z^{3}}{3} - (x^{2}+y^{2})(1+e\,z) + f\,z\,x^{3}$"
	fig.text(0.02, 0.98, line1, color='white', ha='left', va='top', fontsize=10)
	fig.text(0.02, 0.955, line2, color='white', ha='left', va='top', fontsize=10)
	fig.text(0.02, 0.93, line3, color='white', ha='left', va='top', fontsize=10)

	# Slow auto-rotation of the camera around the scene
	rot_speed_deg_per_frame = 0.2  # smaller -> slower rotation
	start_elev, start_azim = ax.elev, ax.azim

	def update_rotation(frame_idx):
		azim = start_azim + rot_speed_deg_per_frame * float(frame_idx)
		ax.view_init(elev=start_elev, azim=azim)
		return ()

	ani = FuncAnimation(fig, update_rotation, frames=np.arange(0, 100000),
				interval=50, blit=False, repeat=True)

	# Tight layout for clean view
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	# Each axis total length = 10,000 (from -5,000 to +5,000)
	draw_xyz_axes(axis_half_length=5000.0)


