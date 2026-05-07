# DEFINE CLASS SOLUTION
#
# An Individual is a whole painting (a list of N_TRIANGLES Triangles).
# A Triangle is one gene block of 10 floats in [0, 1]:
#     (x1, y1, x2, y2, x3, y3, r, g, b, a)
# The genome is normalized so that mutation/crossover operate uniformly on
# a single numeric type and bounds repair is a single np.clip(g, 0, 1).
# Decoding to pixel coordinates / 0-255 byte channels happens at render
# time (W-1, H-1, *255), which removes the off-by-one risk of generating
# coordinates equal to W or H.
#
# Public contract used by operators and the GA loop:
#   - repr                          : the genome (list of Triangle)
#   - random_initial_representation : build a random genome
#   - fitness                       : RMSE vs. the target image (cached)
#   - with_repr                     : build a sibling Individual with a new
#                                     genome but the same target, so that
#                                     mutation/crossover can be expressed
#                                     as pure functions returning new
#                                     Individuals.

import random

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

IMG_WIDTH = 300
IMG_HEIGHT = 400
N_TRIANGLES = 100
GENES_PER_TRIANGLE = 10  # x1, y1, x2, y2, x3, y3, r, g, b, a - all in [0, 1]


class Triangle:
    """One triangle: 10 floats in [0, 1]. Decoded on demand."""

    __slots__ = ("repr",)

    def __init__(self, repr=None):
        if repr is None:
            repr = [random.random() for _ in range(GENES_PER_TRIANGLE)]
        # Defensive copy so two Triangles can never share a list.
        self.repr = list(repr)

    def vertices(self, w=IMG_WIDTH, h=IMG_HEIGHT):
        """Return the 3 vertices in pixel space, decoded from [0, 1]."""
        g = self.repr
        return [
            (round(g[0] * (w - 1)), round(g[1] * (h - 1))),
            (round(g[2] * (w - 1)), round(g[3] * (h - 1))),
            (round(g[4] * (w - 1)), round(g[5] * (h - 1))),
        ]

    def color(self):
        """Return the RGBA color as a 4-tuple of bytes (0-255)."""
        g = self.repr
        return (
            round(g[6] * 255),
            round(g[7] * 255),
            round(g[8] * 255),
            round(g[9] * 255),
        )

    def copy(self):
        return Triangle(self.repr)  # __init__ already copies the list

    def __repr__(self):
        return f"Triangle({self.repr})"


class Individual:
    """A candidate painting: N_TRIANGLES Triangles + cached RMSE fitness.

    Parameters
    ----------
    target : np.ndarray
        Target image as an (H, W, 3) array of dtype float32 (or anything
        castable to float32). Stored on the instance so that fitness() and
        with_repr() can be called without re-passing it.
    repr : list[Triangle], optional
        Existing genome. If None, a random one is generated.
    """

    def __init__(self, target: np.ndarray, repr=None):
        self.target = target
        if repr is None:
            repr = self.random_initial_representation()
        # Defensive copy of each Triangle so children of crossover cannot
        # share Triangle instances with their parents.
        self.repr = [t.copy() for t in repr]
        self._fitness = None  # populated on first fitness() call

    def random_initial_representation(self):
        return [Triangle() for _ in range(N_TRIANGLES)]

    def with_repr(self, new_repr):
        """Build a new Individual that carries the same target image but a
        different genome. Mutation and crossover MUST go through this
        method so that _fitness is left as None on the child and gets
        recomputed from the new genome."""
        return Individual(target=self.target, repr=new_repr)

    def render(self) -> Image.Image:
        """Rasterize the genome to an RGB PIL image.

        Uses one RGBA canvas and ImageDraw.polygon in RGBA mode, which
        alpha-blends each polygon's fill against the existing canvas in a
        single pass - no per-triangle layer is allocated.
        """
        canvas = Image.new("RGBA", (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 255))
        draw = ImageDraw.Draw(canvas, "RGBA")
        for triangle in self.repr:
            draw.polygon(triangle.vertices(), fill=triangle.color())
        return canvas.convert("RGB")

    def fitness(self) -> float:
        """Pixel-wise RMSE between rendered phenotype and the target.
        Cached in self._fitness because rasterization dominates runtime."""
        if self._fitness is not None:
            return self._fitness
        rendered = np.asarray(self.render(), dtype=np.float32)
        diff = rendered - self.target.astype(np.float32, copy=False)
        self._fitness = float(np.sqrt(np.mean(diff * diff)))
        return self._fitness

    def plot(self, ax=None, title="Solution") -> None:
        """Display the rendered phenotype with matplotlib."""
        show = ax is None
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(3, 4))
        ax.imshow(self.render())
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if show:
            plt.tight_layout()
            plt.show()

    def __repr__(self) -> str:
        return str(self.repr)
