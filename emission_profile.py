import numpy as np
import math
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
    lookVectors = [unp.uarray([-0.36734736, 0.78436053, 0.49983439],
                              [0.00000001, 0.00000001, 0.00000001]),
                   unp.uarray([-0.33040446, 0.77870637, 0.53333777],
                              [0.00000001, 0.00000001, 0.00000001]),
                   unp.uarray([-0.29262921, 0.77109057, 0.56549758],
                              [0.00000001, 0.00000001, 0.00000001]),]

    # Simulates an emission profile using a normal distribution
    emission = []
    emissionAltitudes = []
    for i in range(350000, 0, -5000):
        emission.append(250000 / (50000 * math.sqrt(2 * math.pi)) * \
                         math.exp(-1 / 2 * (((i - 2500) - 300000) / 50000) ** 2))
        emissionAltitudes.append(i / 1000)
    
    # Generates the image and sets up its y-axis labels
    image = generateImage(positions[0], lookVectors, emission)
    numPixels = 256
    yAxisValues = []
    for i in image[:, 1][0: numPixels: numPixels // 4]:
        # Converts altitude to km with three decimal places
        altitude = int(i) / 1000
        yAxisValues.append(altitude)
    # Converts altitude to km with three decimal places
    altitude = int(image[numPixels - 1, 1]) / 1000
    yAxisValues.append(altitude)
    yAxisLocations = []
    for i in range(0, numPixels, numPixels // 4):
        yAxisLocations.append(i)
    yAxisLocations.append(numPixels - 1)

    imgFig, imgAx = plt.subplots()
    imgAx.imshow(np.delete(image, 1, axis=1), cmap="afmhot", aspect=0.02)
    imgAx.set_xticks([])
    imgAx.set_yticks(yAxisLocations, yAxisValues)
    imgAx.set_title("Generated Image from Simulated Emission Profile")
    imgAx.set_ylabel("Tangent Point Altitude (km)")
    imgFig.savefig("research/mighti-practice/graphs/Simulated Image (300km Peak).png")

    emFig, emAx = plt.subplots()
    emAx.set_title("Simulated Emission Profile")
    emAx.set_xlabel("Emission (modeled after a photons/cm^3/s emission graph)")
    emAx.set_ylabel("Altitude (km)")
    emAx.plot(emission, emissionAltitudes)
    emFig.savefig("research/mighti-practice/graphs/Simulated Emission Profile (300km Peak).png")

    profile = generateEmissionProfile(positions[0], lookVectors, np.delete(image, 1, axis=1))
    
    fig1, ax1 = plt.subplots()
    ax1.set_title("Emission Profile from Image")
    ax1.set_xlabel("Emission")
    ax1.set_ylabel("Altitude (km)")
    ax1.plot(profile[:, 0], profile[:, 1] / 1000)
    fig1.savefig("research/mighti-practice/graphs/Simulated Emission Profile from Image (300km Peak).png")

    # distances = getLayerDistances(positions[0], lookVectors[0], 5000, 300000)
    # distancesValues = [[], []]
    # distancesUncertainties = [[], []]
    # for i in range(0, len(distances[0])):
    #     distancesValues[0].append(distances[0][i].n / 1000)
    #     distancesValues[1].append(distances[1][i] / 1000)
    #     distancesUncertainties[0].append(distances[0][i].s / 1000)
    #     distancesUncertainties[1].append(distances[1][i] / 1000)
    
    # fig1, ax1 = plt.subplots()
    # ax1.set_title("Ray distances per altitude layer (5km increments)")
    # ax1.set_xlabel("Distance ray passes through layer (km)")
    # ax1.set_ylabel("Layer upper altitude (km)")
    # ax1.plot(distancesValues[0], distancesValues[1])
    # fig1.savefig("research/mighti-practice/graphs/Ray distances per altitude.png")

    # fig2, ax2 = plt.subplots()
    # ax2.set_title("Ray distance uncertainties per altitude layer (5km increments)")
    # ax2.set_xlabel("Ray distance uncertainty (km)")
    # ax2.set_ylabel("Layer upper altitude (km)")
    # ax2.plot(distancesUncertainties[0], distancesUncertainties[1])
    # fig2.savefig("research/mighti-practice/graphs/Ray distance uncertainties per altitude.png")

def generateEmissionProfile(pos, lookVectors, image):
    """Generates an image of the given airglow emission as a 256x1 matrix of floats

    Parameters:
    -----------
    pos : uarray
        The spacecraft position
    lookVectors : list
        A list of the three unit look vectors, ordered with the first being the one pointing
        closest towards the Earth and the last being the one pointed furthest away from
        the Earth
    image : matrix
        The image array, where the first values are the brightness of the top pixels and the
        last being the brightness of the bottom pixels. Each array element is one pixel.
        The matrix's shape is 256x1 and contains only floats.
    
    Returns:
    --------
    matrix
        The computed emission profile from the image. The first column contains the emission
        values, and the second column contains the corresponding altitude values.
    """

    verticalResolution = 256
    layerThickness = 5000
    maxAltitude = 350000

    matrixUncertainties = generateMatrix(pos, lookVectors, verticalResolution, layerThickness,
                                         maxAltitude)
    matrix = np.zeros(np.shape(matrixUncertainties))
    for i in range(0, np.shape(matrixUncertainties)[0]):
        for j in range(0, np.shape(matrixUncertainties)[1]):
            matrix[i, j] = matrixUncertainties[i, j].n
    
    # Generates the emission profile from the image using the matrix
    emissionProfile = np.linalg.pinv(matrix) @ image

    # Adds the corresponding altitude values into the matrix
    altitudes = np.zeros(np.shape(emissionProfile))
    for i in range(0, np.shape(altitudes)[0]):
        altitudes[i, 0] = maxAltitude - layerThickness / 2 - layerThickness * i
    emissionProfile = np.concatenate((emissionProfile, altitudes), axis=1)

    return emissionProfile

def generateImage(pos, lookVectors, emission):
    """Generates an image of the given airglow emission as a 256x1 matrix of floats

    Parameters:
    -----------
    pos : uarray
        The spacecraft position
    lookVectors : list
        A list of the three unit look vectors, ordered with the first being the one pointing
        closest towards the Earth and the last being the one pointed furthest away from
        the Earth
    emission : list
        A list of floats containing brightness values for each atmosphere layer, with the first
        value being the emission at the 350-345km layer, the second at the 345-34s0km layer, and
        so on in 5km layer increments
    
    Returns:
    --------
    matrix
        The image and the tangent point altitude for each pixel. The first column contains the
        image emission data, and the second contains the tangent point altitudes. The first
        rows are the top pixels and the last are the bottom pixels. Each array element is one
        pixel. The matrix's shape is 256x2.
    """

    verticalResolution = 256
    layerThickness = 5000
    maxAltitude = 350000

    matrix = generateMatrix(pos, lookVectors, verticalResolution, layerThickness, maxAltitude)

    # Converts the emission data to a numpy column vector
    numLayers = np.shape(matrix)[1]
    emissionVector = np.zeros((numLayers, 1))
    for i in range(0, numLayers):
        emissionVector[i, 0] = emission[i]
    
    # Generates the image using the matrix and emmission data
    imageWithUncertainties = matrix @ emissionVector
    image = np.zeros(np.shape(imageWithUncertainties))
    for i in range(0, verticalResolution):
        image[i, 0] = imageWithUncertainties[i, 0].n
    
    # Adds tangent point altitudes to the image matrix
    pixelLookVectors = lookInterpolate(lookVectors, verticalResolution)
    tangentAltitudes = np.zeros((verticalResolution, 1))
    for i in range(0, len(pixelLookVectors)):
        tangentPointECEF = getTangentPoint(pos, pixelLookVectors[i])
        tangentPointLLA = ECEFtoLLA(tangentPointECEF)
        tangentPointAltitude = tangentPointLLA[2].n
        tangentAltitudes[-i - 1, 0] = tangentPointAltitude

    return np.concatenate((image, tangentAltitudes), axis=1)

def generateMatrix(pos, lookVectors, verticalResolution, layerThickness, maxAltitude):
    """Generates the look vector/layer distances matrix
    
    Parameters:
    -----------
    pos : uarray
        The spacecraft position
    lookVectors : list
        A list of the three unit look vectors, ordered with the first being the one pointing
        closest towards the Earth and the last being the one pointed furthest away from
        the Earth
    verticalResolution : int
        The number of pixels in the image's vertical column, each with its own look vector.
        Must be even.
    layerThickness : int
        The height of each altitude layer in meters
    maxAltitude : int
        The altitude of the top of the highest layer to use


    Returns:
    --------
    umatrix
        The look vector/layer distances matrix. Each row is for a look vector, and each column
        is for a layer. matrix[i][j] is the distance look vector i's ray passes through altitude
        layer j. j = 0 is the maxAltitude layer, j=1 is the maxAltitude - layerThickness layer,
        and so on.
    """
    
    pixelLookVectors = lookInterpolate(lookVectors, verticalResolution)

    # Creates the empty matrix to be filled
    bottomLayerDistances = getLayerDistances(pos, pixelLookVectors[0], layerThickness, maxAltitude)
    bottomLayerDistances = bottomLayerDistances[0]
    numLayers = len(bottomLayerDistances)
    matrix = unp.umatrix(np.zeros((verticalResolution, numLayers)),
                         np.zeros((verticalResolution, numLayers)))

    # Fills the matrix with data
    for i in range(1, verticalResolution):

        layerDistances = getLayerDistances(pos, pixelLookVectors[i], layerThickness, maxAltitude)
        layerDistances = layerDistances[0]

        while len(layerDistances) < numLayers:
            layerDistances.append(ufloat(0, 0))

        matrix[-1 - i] = matrix[-1 - i] + layerDistances

    matrix[-1] = matrix[-1] + bottomLayerDistances

    return matrix

def lookInterpolate(lookVectors, pixels):
    """
    Interpolates between three look vectors to get the look vector for each pixel in the image

    Parameters:
    -----------
    lookVectors : list
        A list of three uarrays, each containing one of the three unit look vectors
    pixels : int
        The number of pixels in the image to interpolate the look vectors between. Must be even
    
    Returns:
    --------
    list
        The list of interpolated look vector for each pixel, as a list of uarrays
    """
    
    interpolatedLookVectors = []
    halfPixels = pixels / 2
    for i in np.arange(0.5, halfPixels):
        interpolatedLookVectors.append(lookVectors[0] * (1 - i / halfPixels) +
                                       lookVectors[1] * i / halfPixels)
    for i in np.arange(0.5, halfPixels):
        interpolatedLookVectors.append(lookVectors[1] * (1 - i / halfPixels) +
                                       lookVectors[2] * i / halfPixels)

    return interpolatedLookVectors

def getLayerDistances(pos, look, layerThickness, maxAltitude):
    """Returns the distances the given look vector passes through the Earth's atmosphere.

    Parameters
    ----------
    pos : uarray
        The position the look vector originates from (spacecraft position) in ECEF (meters)
    look : uarray
        The look vector in ECEF (meters)
    layerThickness : int
        The height of each altitude layer in meters
    maxAltitude : int
        The altitude of the top of the highest layer to use

    Returns:
    --------
    list
        A list containing two lists. The first is the distance the look vector passes
        through each layer, as ufloats. The second is the upper altitude of each corresponding
        layer, as floats.
    """

    # Finds the distance from the center of the Earth to the tangent point
    tangentPoint = getTangentPoint(pos, look)
    tangentPoint[2] *= equatorRadius / polarRadius
    tangentRadialDistance = unorm(tangentPoint)
    tangentAltitude = ECEFtoLLA(tangentPoint)[2]

    # Scaling the position and look vectors' z-axis to treat Earth's ellipsoid as a sphere
    pos = np.ndarray.copy(pos)
    pos[2] *= equatorRadius / polarRadius
    look = np.ndarray.copy(look)
    look[2] *= equatorRadius / polarRadius

    #TODO remove (plots the intersection points in 3D)
    # points = [[], [], [], []]

    # Calculates the array of distances the ray travels through each layer,
    # below the intersection points for the layer's altitude. Layers go from
    # maxAltitude down to the intersection point altitude. The first list in
    # layerDistances contains the distances, and the second contains the
    # corresponding altitudes
    layerDistances = [[], []]
    currentAltitude = maxAltitude
    a = np.dot(look, look)
    b = 2 * np.dot(look, pos)
    for radialDistance in np.arange(maxAltitude + equatorRadius.n,
            tangentRadialDistance.n, -layerThickness):
        
        # Adds the uncertainty back into radialDistance
        radialDistance = ufloat(radialDistance, equatorRadius.s)

        # Finds the two intersection points between the look vector and this altitude's ellipsoid
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
        layerDistances[0].append(distance)
        layerDistances[1].append(currentAltitude)

        # Tracks the altitude of the current ray layer
        currentAltitude -= layerThickness

        #TODO remove (plots the intersection points in 3D)
        # points[0].append(point1[0].n)
        # points[1].append(point1[1].n)
        # points[2].append(point1[2].n)
        # points[3].append([1, 0, 0])
        # points[0].append(point2[0].n)
        # points[1].append(point2[1].n)
        # points[2].append(point2[2].n)
        # points[3].append([1, 0, 0])

    # Subtracts the distance the ray travels through the lower layer from the distance it
    # travels through the current layer, so each distance doesn't also count the distance of
    # the ray passing through all the layers below this layer
    for i in range(0, len(layerDistances[0]) - 1):
        layerDistances[0][i] -= layerDistances[0][i + 1]

    #TODO remove (plots the intersection points in 3D)
    # for lat in range(-90, 91, 5):
    #     for lon in range(0, 360, 5):
    #         point = LLAtoECEF([ufloat(lat, 0), ufloat(lon, 0), ufloat(0, 0)])
    #         points[0].append(point[0].n)
    #         points[1].append(point[1].n)
    #         points[2].append(point[2].n)
    #         points[3].append([0.1, 0.4, 1])
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(projection='3d')
    # ax3.set_box_aspect((np.ptp(points[0]), np.ptp(points[1]), np.ptp(points[2])))
    # ax3.scatter(points[0], points[1], points[2], c=points[3])
    # fig3.savefig("research/mighti-practice/Intersection Points.png")
    # fig3.show(True)

    return layerDistances

def getTangentPoint(pos, look):
    """Finds a look vector's tangent point to Earth's surface

    Parameters:
    -----------
    pos : uarray
        The position the look vector originates from (spacecraft position) in ECEF (meters)
    look : uarray
        The look vector in ECEF (meters)

    Returns:
    --------
    uarray
        The tangent point in ECEF (meters)
    """

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

def ECEFtoLLA(pos):
    """Converts ECEF coordinates (meters) to LLA coordinates (meters)

    Parameters:
    -----------
    pos : uarray
        The position in ECEF (meters) to convert

    Returns:
    --------
    uarray
        The position in LLA (meters)
    """

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

def LLAtoECEF(pos):
    """Converts LLA coordinates (meters) to ECEF coordinates (meters)

    Parameters:
    -----------
    pos : uarray
        The position in LLA (meters) to convert

    Returns:
    --------
    uarray
        The position in ECEF (meters)
    """

    posECEF = unp.uarray([0, 0, 0], [0, 0, 0])
    deg2rad = math.pi / 180
    N = equatorRadius / umath.sqrt(1 - e ** 2 * umath.sin(pos[0] * deg2rad) ** 2)
    posECEF[0] = (N + pos[2]) * umath.cos(pos[0] * deg2rad) * umath.cos(pos[1] * deg2rad)
    posECEF[1] = (N + pos[2]) * umath.cos(pos[0] * deg2rad) * umath.sin(pos[1] * deg2rad)
    posECEF[2] = (N * polarRadius ** 2 / equatorRadius ** 2 + pos[2]) * \
            umath.sin(pos[0] * deg2rad)
    return posECEF

def unorm(arr):
    """Returns the norm of an array of ufloats

    Parameters:
    -----------
    arr : uarray
        The uarray to get the norm of

    Returns:
    --------
    ufloat
        The norm of the given uarray
    """

    sumOfSquares = 0
    for element in arr:
        sumOfSquares += element ** 2
    return umath.sqrt(sumOfSquares)

main()