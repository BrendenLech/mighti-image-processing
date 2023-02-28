import math
import numpy as np

# Earth reference ellipsoid WGS84 parameters
a = 6378137
f = 1 / 298.257223563
b = a * (1 - f)
e = math.sqrt((a ** 2 - b ** 2) / a ** 2)
ePrime = math.sqrt((a ** 2 - b ** 2) / b ** 2)

# Earth radii
equatorRadius = 6378.14
polarRadius = 6356.75

# Main function definition
def main():

    # Given spacecraft position, look vectors, and correct tangent points, in ECEF (m)
    positions = [np.array([5657188.6, -2642459.2, 3071963.2])]
    lookVectors = [np.array([-0.36734736, 0.78436053, 0.49983439]),
                   np.array([-0.33040446, 0.77870637, 0.53333777]),
                   np.array([-0.29262921, 0.77109057, 0.56549758])]
    correctTangentPoints = [np.array([4701623.3, -602135.07, 4372161.4]),
                            np.array([4905927.9, -871867.15, 4284645.8]),
                            np.array([5089290.9, -1146024.3, 4169409.3])]
    index = 0
    
    # Prints given position and look vectors in ECEF (km)
    print(f'\nPos (ECEF): {positions[0] / 1000}')
    print(f'Look (ECEF): {lookVectors[index]}')

    # Prints the calculated tangent point and the error from the correct value in ECEF (km)
    print(f'Tangent Point (ECEF): {getTangentPoint(positions[0], lookVectors[index]) / 1000}')
    print(f'Correct Tangent Point (ECEF): {correctTangentPoints[index] / 1000}')
    tangentPointError = (getTangentPoint(positions[0], lookVectors[index])
        - correctTangentPoints[index])
    tangentPointError = np.linalg.norm(tangentPointError) / 1000
    print(f'Tangent Point Difference (ECEF): {tangentPointError}')

    # Prints the calculated tangent point's angle with the look vector
    tangentPoint = getTangentPoint(positions[0], lookVectors[index])
    tpMag = np.linalg.norm(tangentPoint)
    lookMag = np.linalg.norm(lookVectors[index])
    dot = np.dot(lookVectors[index], tangentPoint)
    print(f'Tangent point angle: {math.degrees(math.acos(dot / (tpMag * lookMag)))}')

    # Prints the position and tangent point in LLA (m)
    print(f'\nPos (LLA): {ECEFtoLLA(positions[0])}')
    print(f'Tangent Point (LLA): {ECEFtoLLA(getTangentPoint(positions[0], lookVectors[index]))}')

# Takes spacecraft position and look vectors in ECEF coordinates (meters) and finds the look
# vector's tangent point to Earth's ellipsoid in ECEF (meters)
def getTangentPoint(pos, look):

    # Scaling the z-axis to treat Earth's ellipsoid as a sphere
    pos = np.ndarray.copy(pos)
    pos[2] *= equatorRadius / polarRadius
    look = np.ndarray.copy(look)
    look[2] *= equatorRadius / polarRadius

    # Finding the tangent point
    f = -np.dot(pos, look) / np.dot(look, look)
    tangentPoint = pos + f * look

    # Scaling back to ECEF coordinates
    tangentPoint[2] *= polarRadius / equatorRadius
    return tangentPoint

# Converts ECEF coordinates (meters) to LLA coordinates (meters)
def ECEFtoLLA(pos):

    p = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
    theta = math.atan(pos[2] * a / (p * b))

    latitude = math.atan((pos[2] + ePrime ** 2 * b * math.sin(theta) ** 3) /
        (p - e ** 2 * a * math.cos(theta) ** 3))
    longitude = math.atan(pos[1] / pos[0])
    N = a / math.sqrt(1 - e ** 2 * math.sin(latitude) ** 2)
    altitude = p / math.cos(latitude) - N

    latitude = math.degrees(latitude)
    longitude = math.degrees(longitude)
    if longitude < 360:
        longitude += 360

    return np.array([latitude, longitude, altitude])


main()