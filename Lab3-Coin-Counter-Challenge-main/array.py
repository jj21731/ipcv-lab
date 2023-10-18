# Example code for allocating a 3D int array on the heap, 
# many other more efficient solutions are possible

# import numpy as np
#
# dim1 = 10
# dim2 = 20
# dim3 = 30
#
# ipppArr = np.zeros([dim1, dim2, dim3], dtype=np.float32)
#
# for i in range(0, dim1):
#     for j in range(0, dim2):
#         for k in range(0, dim3):
#             ipppArr[i,j,k] = i * j * k
#
# for i in range(0, dim1):
#     for j in range(0, dim2):
#         for k in range(0, dim3):
#             print("[%d]" % ipppArr[i,j,k])
