'''
    This file is part of the September 2018 Workshop at Yuan Ze University.

    You can use these examples in the way you seem fit, though I can't make sure
    it will work fine in your case.
'''

# tensor product
from numpy import array
from numpy import tensordot
A = array([1, 2])
B = array([3, 4])
C = tensordot(A, B, axes=0)

print("\nResult is: \n")
print(C.shape)
print(C)

# more complex
A = array([
  [[1, 2, 3],    [4, 5, 6],    [7, 8, 9]],
  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
  [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
  ])

B = array([
  [[1, 2, 3],    [4, 5, 6],    [7, 8, 9]],
  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
  [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
  ])

C = tensordot(A, B, axes=0)

print("\nResult is: \n")
print(C.shape)
print(C)
