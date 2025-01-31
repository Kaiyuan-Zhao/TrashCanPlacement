import cv2
import numpy as np

def pixToNum(pixel, colourMap):
    for color, number in colourMap.items():
        if np.array_equal(pixel, np.array(color)):
            return number
    return -1 #if no recognized colour


def translate(imPath, color_mapping, output_file):
    image = cv2.imread(imPath, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image read error")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    print("height: ", height, "width: ", width)

    with open(output_file, "w") as f:
        for y in range(height):
            row = []
            for x in range(width):
                row.append(str(pixToNum(image[y, x], color_mapping)))
            f.write(" ".join(row) + "\n")


def runConv():
    imPath = "C:\\Users\\kihoi\\Desktop\\smap_1.png"  # Change this to the path of your image
    output = "C:\\Users\\kihoi\\Desktop\\output.txt"  # Output text file

    #colours
    colours = {
        (255, 255, 255): 0,  #white - empty for trash cans
        (0, 0, 0): 1,  #black - wall
        (255, 0, 0): 2,  #red - destination
        (0, 0, 255): 3  #blue - spawn
    }

    translate(imPath, colours, output)
    print(f"Output saved to {output}")


runConv()
