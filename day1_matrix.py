"""
DAY 1: Linear Algebra Essentials
Faran Ali - (Matrix Multiplication from Scratch)
"""

import numpy as np

print("=" * 50)
print("DAY 1: MATRIX-VECTOR MULTIPLICATION")
print("=" * 50)

# EXAMPLE: The one from the video
print("\n--- EXAMPLE: Video Example ---")
A = np.array([[2, 1],
              [1, 3]])
v = np.array([1, 2])

result = A @ v
print(f"A = \n{A}")
print(f"v = {v}")
print(f"A @ v = {result}")
print(f"Expected: [4 7]")
print(f"Correct? {result[0] == 4 and result[1] == 7}")

print("\n" + "=" * 50)
print("EXERCISE 1: Identity Matrix")
print("=" * 50)

# EXERCISE 1: Identity Matrix (does nothing to the vector)
# An identity matrix has 1's on diagonal, 0's elsewhere
I = np.array([[1, 0],
              [0, 1]])
v1 = np.array([5, 7])

result1 = I @ v1
print(f"Identity Matrix I = \n{I}")
print(f"Vector v = {v1}")
print(f"I @ v = {result1}")
print(f"Expected: [5, 7]")
print(f"Correct? {result1[0] == 5 and result1[1] == 7}")

print("\n" + "=" * 50)
print("EXERCISE 2: Scaling Matrix")
print("=" * 50)

# EXERCISE 2: Scaling Matrix (stretches/shrinks the vector)
# Diagonal matrix with numbers >1 stretches, <1 shrinks
S = np.array([[3, 0],
              [0, 2]])
v2 = np.array([1, 1])

result2 = S @ v2
print(f"Scale Matrix S = \n{S}")
print(f"Vector v = {v2}")
print(f"S @ v = {result2}")
print(f"Expected: [3, 2]")
print(f"Explanation: x was multiplied by 3, y was multiplied by 2")
print(f"Correct? {result2[0] == 3 and result2[1] == 2}")

print("\n" + "=" * 50)
print("EXERCISE 3: Shear Matrix")
print("=" * 50)

# EXERCISE 3: Shear Matrix (slants the vector)
# The top-right value (2) adds to x based on y's value
Sh = np.array([[1, 2],
              [0, 1]])
v3 = np.array([1, 1])

result3 = Sh @ v3
print(f"Shear Matrix Sh = \n{Sh}")
print(f"Vector v = {v3}")
print(f"Sh @ v = {result3}")
print(f"Expected: [3, 1]")
print(f"Explanation: x became (1*1 + 2*1) = 3, y stayed 1")
print(f"Correct? {result3[0] == 3 and result3[1] == 1}")

print("\n" + "=" * 50)
print("EXERCISE 4: Rotation Matrix (90 degrees)")
print("=" * 50)

# EXERCISE 4: Rotation Matrix (rotates vector 90 degrees counter-clockwise)
# This matrix sends (1,0) to (0,1) and (0,1) to (-1,0)
R = np.array([[0, -1],
              [1, 0]])
v4 = np.array([1, 0])

result4 = R @ v4
print(f"Rotation Matrix R = \n{R}")
print(f"Vector v = {v4}")
print(f"R @ v = {result4}")
print(f"Expected: [0, 1]")
print(f"Explanation: A vector pointing right (1,0) rotates to pointing up (0,1)")
print(f"Correct? {result4[0] == 0 and result4[1] == 1}")

print("\n" + "=" * 50)
print("EXERCISE 5: Projection Matrix (onto x-axis)")
print("=" * 50)

# EXERCISE 5: Projection Matrix (flattens vector onto x-axis)
# Keeps x, sets y to 0
P = np.array([[1, 0],
              [0, 0]])
v5 = np.array([4, 9])

result5 = P @ v5
print(f"Projection Matrix P = \n{P}")
print(f"Vector v = {v5}")
print(f"P @ v = {result5}")
print(f"Expected: [4, 0]")
print(f"Explanation: x stayed 4, y became 0 (flattened onto x-axis)")
print(f"Correct? {result5[0] == 4 and result5[1] == 0}")

print("\n" + "=" * 50)
print("BONUS: Try your own!")
print("=" * 50)

# BONUS: Create your own matrix and vector
print("\nTry changing these numbers:")
print("Matrix M = [[2, 0], [0, 0.5]]  # Stretch x by 2, shrink y by half")
M = np.array([[2, 0],
              [0, 0.5]])
w = np.array([3, 4])

result6 = M @ w
print(f"M = \n{M}")
print(f"w = {w}")
print(f"M @ w = {result6}")
print(f"Calculation: x: 2*3 + 0*4 = 6, y: 0*3 + 0.5*4 = 2")

print("\n" + "=" * 50)
print("WHAT I LEARNED TODAY")
print("=" * 50)

print("""
1. A matrix is a transformation that changes vectors
2. Matrix @ Vector = transformed vector
3. Different matrices do different things:
   - Identity: leaves vector alone
   - Diagonal: scales each dimension
   - Shear: adds one coordinate to another
   - Rotation: turns vectors around origin
   - Projection: flattens onto a line/plane
4. The columns of a matrix tell you where the basis vectors go
""")

print("\n✅ DAY 1 COMPLETE!")# Should be [4, 0]