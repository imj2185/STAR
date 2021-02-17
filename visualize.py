import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

parts = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
				(6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
				(11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
				(16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
				(22, 23), (21, 21), (23, 8), (24, 25), (25, 12)]

def skeleton_visual(skeletons):
	fig= plt.figure()
	ax = plt.axes(projection='3d')
	for i in parts:
		start_joint = i[0]-1
		end_joint = i[1]-1
		VecStart_x = skeletons[start_joint][0]
		VecEnd_x = skeletons[end_joint][0]
		VecStart_y = skeletons[start_joint][1]
		VecEnd_y =  skeletons[end_joint][1]
		VecStart_z = skeletons[start_joint][2]
		VecEnd_z =  skeletons[end_joint][2]
		ax.plot(xs=[VecStart_x, VecEnd_x], ys=[VecStart_y,VecEnd_y],zs=[VecStart_z,VecEnd_z])
	#x,y,z = [1, 0, 0],[2, 0, 0],[3, 0, 0]
	#sc = ax.scatter(x, y, z, s=40)
	#ax.plot(x,y, color='r')

	plt.show()
	return


def test():
	skeletons, labels = torch.load("dataset/ntu60_cs_val_test.pt")
	#print(skeletons[0].x[0)
	skeleton_visual(skeletons[0].x[70])


if __name__ == "__main__":
	test()

