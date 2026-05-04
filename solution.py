# DEFINE CLASS SOLUTION

from sklearn.metrics import root_mean_squared_error as rmse
import random
import numpy as np
from PIL import Image, ImageDraw

IMG_WIDTH   = 300
IMG_HEIGHT  = 400
N_TRIANGLES = 100


# Remeber an Individual / solution is a whole painting, not a single triangle


class Triangle():
    def __init__(self, repr = None):
        # If no representation is given, a solution is randomly initialized.
        if repr is None:
            repr = self.random_initial_representation()

        self.repr = repr

    def random_initial_representation(self) -> list:
        # 1 triangle is represented by 6 integers: x1, y1, x2, y2, x3, y3 corresponding to the coordinates of the 3 vertices of the triangle
        #  with x between 0 and 300 and y between 0 and 400
        # It will also have rgb values between 0 and 255 and an alpha value between 0 and 1, but we will not include those in the representation, as they are not relevant for the fitness function (we will always set them to the same values)
        # so in total we will have 10 integers for the representation of a triangle

        x1 = random.randint(0, IMG_WIDTH)
        y1 = random.randint(0, IMG_HEIGHT)
        x2 = random.randint(0, IMG_WIDTH)
        y2 = random.randint(0, IMG_HEIGHT)
        x3 = random.randint(0, IMG_WIDTH)
        y3 = random.randint(0, IMG_HEIGHT)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        alpha = random.random()  # float between 0 and 1

        return [x1, y1, x2, y2, x3, y3, r, g, b, alpha]
    

    @property
    def vertices(self) -> list:
        r = self.repr
        return [(r[0], r[1]), (r[2], r[3]), (r[4], r[5])]
    
    @property
    def color(self) -> tuple:
        r = self.repr
        return (r[6], r[7], r[8], int(r[9] * 255)) # convert alpha to 0-255 range for RGBA, PIL expects that



class Individual():

    def __init__(self, repr = None):
        # If no representation is given, a solution is randomly initialized.
        if repr is None:
            repr = self.random_initial_representation()

        self._fitness = None  # Cache fitness value, to avoid recomputation
        self.repr = repr
    
    def random_initial_representation(self) -> list:
        return [Triangle() for _ in range(N_TRIANGLES)]
    
    def render(self) -> Image.Image:
        # Render the solution as a PIL RGBA image, composited onto black.
        # Black background in RGBA so alpha blending works
        canvas = Image.new("RGBA", (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 255)) # creates a complete opac and black canvas

        for triangle in self.repr:
            # Temporary layer for each triangle (transparent background)
            layer = Image.new("RGBA", (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))
            draw  = ImageDraw.Draw(layer)

            # Draw the triangle on the layer
            draw.polygon(triangle.vertices, fill=triangle.color)

            # Pastes the triangle layer onto the canvas using alpha compositing
            canvas = Image.alpha_composite(canvas, layer)

        return canvas.convert("RGB")  # return RGB for fitness computation
    
    def fitness(self, target: np.ndarray) -> float:
        # Calculate rmse between the solution and the target
        rendered = np.array(self.render(), dtype=np.float32)  # (400, 300, 3)
        rmse = np.sqrt(np.mean((rendered - target) ** 2))
        self.fitness = float(rmse)

        return self.fitness

        
    # Method that is called when we run: print(object_of_the_class)
    def __repr__(self) -> str:
        return str( self.repr)