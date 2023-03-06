import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import umath
from uncertainties import unumpy as unp

# Earth reference ellipsoid WGS84 parameters
equatorRadius = ufloat(6378137, 1)
f = 1 / ufloat(298.257223563, 0.000000001)
polarRadius = equatorRadius * (1 - f)
e = umath.sqrt((equatorRadius ** 2 - polarRadius ** 2) / equatorRadius ** 2)
ePrime = umath.sqrt((equatorRadius ** 2 - polarRadius ** 2) / polarRadius ** 2)

# Main function definition
def main():

    # Given spacecraft position, look vectors, and correct tangent points, in ECEF (m)
    positions = [unp.uarray([5657188.6, -2642459.2, 3071963.2], [0.1, 0.1, 0.1])]
    lookVectors = [unp.uarray([0.36734736, 0.78436053, 0.49983439],
                              [0.00000001, 0.00000001, 0.00000001]),
                   unp.uarray([-0.33040446, 0.77870637, 0.53333777],
                              [0.00000001, 0.00000001, 0.00000001]),
                   unp.uarray([-0.29262921, 0.77109057, 0.56549758],
                              [0.00000001, 0.00000001, 0.00000001]),]

    distances = getLayerDistances(positions[0], lookVectors[2], 5000, 300000)
    distancesValues = [[], []]
    distancesUncertainties = [[], []]
    for entry in distances:
        distancesValues[0].append(entry[0].n / 1000)
        distancesValues[1].append(entry[1].n / 1000)
        distancesUncertainties[0].append(entry[0].n / 1000)
        distancesUncertainties[1].append(entry[1].s / 1000)
    
    fig1, ax1 = plt.subplots()
    ax1.set_title("Ray distances per altitude layer (5km increments)")
    ax1.set_xlabel("Distance ray passes through layer (km)")
    ax1.set_ylabel("Distance from center of Earth (km)")
    ax1.plot(distancesValues[1], distancesValues[0])
    fig1.savefig("research/mighti-practice/Ray distances per altitude.png")

    fig2, ax2 = plt.subplots()
    ax2.set_title("Ray distance uncertainties per altitude layer (5km increments)")
    ax2.set_xlabel("Ray distance uncertainty (km)")
    ax2.set_ylabel("Distance from center of Earth (km)")
    ax2.plot(distancesUncertainties[1], distancesUncertainties[0])
    fig2.savefig("research/mighti-practice/Ray distance uncertainties per altitude.png")

# Returns the distances the given look vector passes through the Earth's atmosphere in,
# in increments of 5km altitude from the tangent point to 300km
def getLayerDistances(pos, look, increment, maxAltitude):

    # Finds the distance from the center of the Earth to the tangent point
    tangentPoint = getTangentPoint(pos, look)
    tangentPoint[2] *= equatorRadius / polarRadius
    tangentRadialDistance = unorm(tangentPoint)

    # Scaling the position and look vectors' z-axis to treat Earth's ellipsoid as a sphere
    pos = np.ndarray.copy(pos)
    pos[2] *= equatorRadius / polarRadius
    look = np.ndarray.copy(look)
    look[2] *= equatorRadius / polarRadius

    # Calculates the array of distances below the intersection points for the given altitude
    # increments
    layerDistances = []
    for radialDistance in np.arange(tangentRadialDistance.n + increment,
            tangentRadialDistance.n + maxAltitude, increment):
        
        # Includes the uncertainty back into the radial distance value
        radialDistance = ufloat(radialDistance, tangentRadialDistance.s)

        # Finds the two intersection points between the look vector and this altitude's ellipsoid
        a = np.dot(look, look)
        b = 2 * np.dot(look, pos)
        c = np.dot(pos, pos) - radialDistance ** 2
        discriminant = b ** 2 - 4 * a * c
        f1 = (-b + umath.sqrt(discriminant)) / (2 * a)
        f2 = (-b - umath.sqrt(discriminant)) / (2 * a)
        point1 = pos + f1 * look
        point2 = pos + f2 * look

        # Converts the intersection points back from ECEF' to ECEF
        point1[2] *= polarRadius / equatorRadius
        point2[2] *= polarRadius / equatorRadius
        distance = unorm(point1 - point2)
        layerDistances.append(np.array([radialDistance, distance]))

    # Subtracts the distance the ray travels through the lower layer from the distance it
    # travels through the current layer, so each distance doesn't also count the distance of
    # the ray passing through all the layers below this layer
    for i in range(len(layerDistances) - 1, 0, -1):
        layerDistances[i][1] -= layerDistances[i - 1][1]

    return layerDistances

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

    p = umath.sqrt(pos[0] ** 2 + pos[1] ** 2)
    theta = umath.atan(pos[2] * equatorRadius / (p * polarRadius))

    latitude = umath.atan((pos[2] + ePrime ** 2 * polarRadius * umath.sin(theta) ** 3) /
        (p - e ** 2 * equatorRadius * umath.cos(theta) ** 3))
    longitude = umath.atan(pos[1] / pos[0])
    N = equatorRadius / umath.sqrt(1 - e ** 2 * umath.sin(latitude) ** 2)
    altitude = p / umath.cos(latitude) - N

    latitude = umath.degrees(latitude)
    longitude = umath.degrees(longitude)
    if longitude < 360:
        longitude += 360

    return np.array([latitude, longitude, altitude])

# Returns the norm of an array of ufloats
def unorm(arr):
    sumOfSquares = 0
    for element in arr:
        sumOfSquares += element ** 2
    return umath.sqrt(sumOfSquares)


main()