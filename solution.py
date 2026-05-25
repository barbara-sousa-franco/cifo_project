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
# Domain constraints (additions over the baseline encoding):
#   - MAX_TRIANGLE_SIZE limits the bounding-box span of any triangle in
#     normalized [0, 1] coordinates. With 100 triangles and 300x400 pixels,
#     a single triangle that spans the whole canvas wastes the rest of
#     the genome by hiding it underneath. Bounding triangles forces the GA
#     to combine many smaller triangles, which is what enables local detail.
#   - ALPHA_MIN/ALPHA_MAX clip the alpha channel: alpha=0 makes a triangle
#     invisible (wasted gene); alpha=1 makes it fully opaque (hides the
#     triangles below it, defeating the alpha-blending stacking that gives
#     the rendered painting its smooth gradients).
#   - _repair_degenerate_vertices pushes coincident/colinear vertices apart
#     so every triangle has positive pixel area.


import random

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

IMG_WIDTH = 300
IMG_HEIGHT = 400
N_TRIANGLES = 100
GENES_PER_TRIANGLE = 10  # x1, y1, x2, y2, x3, y3, r, g, b, a - all in [0, 1]


# Antialiasing supersampling factor. When AA is on, render() draws on a
# canvas SUPERSAMPLE_FACTOR times larger and then downsamples with LANCZOS.
# Costs O(factor^2) more time per render. Default is opt-in (off) so that
# the hyperparameter sweeps in main.ipynb stay tractable.
SUPERSAMPLE_FACTOR = 2



def _clip01(value: float) -> float:

    """Clip a single float to [0, 1].

    Parameters:
        - value (float): The value to clip.

    Returns:
        - float: value clamped to [0, 1].    
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value





def clip_alpha(value: float, alpha_min: float, alpha_max: float) -> float:

    """Clip the alpha gene to the allowed visibility window.

    Parameters:
        - value (float): Raw alpha value after perturbation.
        - alpha_min (float): Minimum allowed opacity.
        - alpha_max (float): Maximum allowed opacity.

    Returns:
        - float: Alpha clamped to [alpha_min, alpha_max].
    
    """
    if value < alpha_min:
        return alpha_min
    if value > alpha_max:
        return alpha_max
    return value






def shrink_to_max_size(genes: list, max_triangle_size: float) -> list:
    """Contract triangle vertices toward their centroid so the bounding box
    fits within MAX_TRIANGLE_SIZE on each axis.

    A determinstic shrink toward the centroid is preferred over rejection
    sampling because (a) it is O(1) and (b) it preserves the *shape* the
    operator proposed, only rescaling it. Color/alpha channels are untouched.

    Parameters:
        - genes (list): The list of genes representing the triangle.
        - max_triangle_size (float): The maximum allowed size for the triangle.

    Returns:
        - list: A mutated copy of ``genes`` (first 6 entries: vertex coords).

    """
    g = list(genes)
    xs = (g[0], g[2], g[4])
    ys = (g[1], g[3], g[5])
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    # Compute the scale factor needed to bring each axis within the limit.
    # If already within the limit, scale is 1.0 (no shrink needed).
    scale_x = max_triangle_size / span_x if span_x > max_triangle_size else 1.0
    scale_y = max_triangle_size / span_y if span_y > max_triangle_size else 1.0

    # Use the more restrictive of the two axes so both constraints are met.
    scale = min(scale_x, scale_y)

    if scale >= 1.0:
        return g   # already within bounds — nothing to do

    # Contract around the centroid; preserves the triangle's shape and centre.
    cx = (g[0] + g[2] + g[4]) / 3.0
    cy = (g[1] + g[3] + g[5]) / 3.0

    # Pull each vertex toward the centroid by the scale factor.
    # new_vertex = centroid + (old_vertex - centroid) * scale
    # _clip01 guards against floating-point drift outside [0, 1].
    for i in (0, 2, 4):
        g[i] = _clip01(cx + (g[i] - cx) * scale)
        g[i + 1] = _clip01(cy + (g[i + 1] - cy) * scale)

    return g







# --------------------------------------------------------------------------
# Perceptual color distance (Challenge 1)
# --------------------------------------------------------------------------
# RMSE in raw RGB treats every channel difference as equally important and
# is not aligned with how the human visual system perceives color
# differences. The CIE recommends working in the CIE Lab color space and
# measuring color distance with CIEDE2000 (Sharma, Wu, Dalal, "The
# CIEDE2000 color-difference formula: implementation notes, supplementary
# test data, and mathematical observations", Color Research & Application,
# 2005). Lab is approximately perceptually uniform, and CIEDE2000 corrects
# the residual non-uniformities of CIE76 around the blue and gray regions.
#
# Pipeline: sRGB (0..255) -> linear RGB (gamma expansion) -> XYZ (D65) ->
#           CIE Lab -> per-pixel ΔE2000 -> mean over the image.


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB values to linear RGB via inverse gamma expansion (IEC 61966-2-1).

    sRGB values are gamma-compressed for display, this undoes that compression
    so that subsequent colour operations (XYZ conversion, distance metrics) are
    performed in a physically linear light space rather than a perceptual one.

    Parameters:
        - c (np.ndarray): sRGB array with values in [0, 255].

    Returns:
        - np.ndarray: Linear RGB array with values in [0, 1].
    """
    c = c / 255.0
    # Two-piece function defined by the sRGB standard:
    # - below the threshold: linear segment (avoids infinite gradient at 0)
    # - above: power curve with exponent 2.4 (the "gamma")
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image to CIE Lab colour space under D65 illuminant.

    Pipeline: sRGB (0..255) → linear RGB → XYZ (D65) → CIE Lab.
    CIE Lab is approximately perceptually uniform — equal Euclidean distances
    in Lab correspond more closely to equal perceived colour differences than
    equal distances in sRGB, which is why it is used as the basis for CIEDE2000.

    Args:
        rgb (np.ndarray): Image array of shape (H, W, 3) with sRGB values in [0, 255].

    Returns:
        np.ndarray: Lab image of shape (H, W, 3) with channels [L*, a*, b*].
    """
    rgb = np.asarray(rgb, dtype=np.float32)

    # 1. gamma expansion (linear RGB in [0, 1])
    linear = _srgb_to_linear(rgb)

    # 2. Convert linear RGB to CIE XYZ using the standard D65 colour matrix (M).
    # Each row of M converts one RGB channel to the X, Y, Z tristimulus values.
    M = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = linear @ M.T

    # 3. normalise XYZ by the D65 white point so that a perfect white
    # gives (1, 1, 1), required before the Lab nonlinear transform.
    xyz_n = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz = xyz / xyz_n

    # 4. apply the CIE f() function (cube root with linear segment near 0)
    # to each normalised XYZ channel independently.
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3,
                 np.cbrt(xyz),
                 xyz / (3 * delta ** 2) + 4.0 / 29.0)

    # 5. compute L*, a*, b* from the transformed XYZ channels.
    # L* = lightness, a* = green-red axis, b* = blue-yellow axis.
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute the CIEDE2000 perceptual colour difference between two Lab images.

    CIEDE2000 corrects the residual non-uniformities of CIE76 (plain Lab distance)
    around blue and grey regions by introducing hue-rotation, chroma, and lightness
    weighting terms. The result is a per-pixel ΔE value where ΔE ≈ 1 corresponds
    roughly to the just-noticeable difference threshold for the human eye.

    Reference: Sharma, Wu, Dalal, "The CIEDE2000 color-difference formula",
    Color Research & Application, 2005.

    Parameters:
        - lab1 (np.ndarray): First Lab image, shape (H, W, 3).
        - lab2 (np.ndarray): Second Lab image, shape (H, W, 3).

    Returns:
        - np.ndarray: Per-pixel ΔE2000 values, shape (H, W). Non-negative.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # 1. Adjust a* channel for chroma (G factor)
    # The G factor rotates the a* axis to correct for the non-uniformity
    # of CIE76 in the blue region. 1e-12 avoids division by zero.
    C1 = np.sqrt(a1 * a1 + b1 * b1) # Chroma, measures saturation
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25.0 ** 7 + 1e-12)))

    # Adjusted a* values and recomputed chroma C'.
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    # 2. Compute hue angles h' in [0, 360)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    # 3. Compute ΔL', ΔC', ΔH'
    dLp = L2 - L1
    dCp = C2p - C1p

    # Hue difference: wrap to [-180, 180] and handle achromatic case.
    dhp = h2p - h1p
    dhp = np.where(dhp >  180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)  # achromatic: no meaningful hue
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    # 4. Compute mean L', C', h' for weighting functions
    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    # Mean hue: special cases for achromatic colours and wrap-around.
    h_sum  = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    hp_bar = np.where(
        (C1p * C2p) == 0,   # achromatic
        h_sum,
        np.where(
            h_diff <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    # 5. Weighting functions Sl, Sc, Sh and rotation term Rt
    # T modulates the hue weighting as a function of mean hue angle.
    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    # d_theta and Rc define the hue-rotation correction for the blue region.
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Rc  = 2.0 * np.sqrt(Cp_bar ** 7 / (Cp_bar ** 7 + 25.0 ** 7 + 1e-12))
    Sl  = 1 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    Sc  = 1 + 0.045 * Cp_bar
    Sh  = 1 + 0.015 * Cp_bar * T
    Rt  = -np.sin(np.radians(2.0 * d_theta)) * Rc

    # 6. Final ΔE2000
    # Weighted sum of lightness, chroma, and hue differences plus the
    # interaction term Rt that corrects for blue-region non-uniformity.
    dE_sq = (
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    # np.maximum guards against tiny negative values from floating-point error.
    return np.sqrt(np.maximum(dE_sq, 0.0))









class Triangle:
    """One triangle: 10 floats in [0, 1]. Decoded on demand."""

    # __slots__ avoids creating a dynamic __dict__ for each instance.
    # This reduces memory usage and can slightly improve attribute access speed.
    __slots__ = ("repr", "alpha_min", "alpha_max", "max_triangle_size")


    def __init__(self, max_triangle_size = 1, alpha_min = 0, alpha_max = 1, repr=None):

        if repr is None:
            repr = [random.random() for _ in range(GENES_PER_TRIANGLE)]
            repr[9] = alpha_min + random.random() * (alpha_max - alpha_min)

        # Defensive copy so two Triangles can never share a list, and apply
        # the domain constraints once at construction so every Triangle that
        # leaves __init__ already satisfies them.
        genes = [_clip01(float(g)) for g in repr]

        # Alpha (gene 9) clipped to [ALPHA_MIN, ALPHA_MAX]
        genes[9] = clip_alpha(genes[9], alpha_min, alpha_max)

        # Size constraint: shrink toward centroid if bbox exceeds MAX_TRIANGLE_SIZE
        genes = shrink_to_max_size(genes, max_triangle_size)
        self.repr = genes

        # Repair degenerate (zero-area / colinear) triangles
        self._repair_degenerate_vertices()
        self.max_triangle_size = max_triangle_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max




    def _repair_degenerate_vertices(self):
        """Ensure the triangle has three distinct non-collinear pixel vertices.

        A triangle whose vertices collapse to fewer than 3 distinct pixel
        positions has zero area and contributes nothing to the rendered
        image - effectively a wasted gene block. We perturb the vertices
        until the area is positive or fall back to a deterministic tiny triangle
        near (x1, y1). This ensures the triangle is represented by three distinct, non-collinear
        vertices in pixel space.
        """
        # Try several small random perturbations to recover a valid triangle.
        for _ in range(12):
            vertices = self.vertices()

            if len(set(vertices)) == 3:
                (x1, y1), (x2, y2), (x3, y3) = vertices

                # 2 * triangle area, sign-agnostic
                if abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) > 0:
                    return
                

            # If the triangle is still degenerate, slightly perturb each vertex
            # coordinate independently and retry.
            for vertex in range(3):
                x_idx = 2 * vertex
                y_idx = x_idx + 1
                self.repr[x_idx] = _clip01(self.repr[x_idx] + random.uniform(-0.01, 0.01))
                self.repr[y_idx] = _clip01(self.repr[y_idx] + random.uniform(-0.01, 0.01))

        # Deterministic fallback near the first vertex if random nudging failed.
        x = min(0.98, max(0.02, self.repr[0]))
        y = min(0.98, max(0.02, self.repr[1]))
        self.repr[0:6] = [x, y, x + 0.01, y, x, y + 0.01]




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
        new_tri = Triangle.__new__(Triangle)
        new_tri.repr = list(self.repr)
        new_tri.alpha_min = self.alpha_min
        new_tri.alpha_max = self.alpha_max
        new_tri.max_triangle_size = self.max_triangle_size
        return new_tri

    def __repr__(self):
        return f"Triangle({self.repr})"


class Individual:
    """A candidate painting: N_TRIANGLES Triangles + cached fitness.

    Parameters:

        - target (np.ndarray): Target image as an (H, W, 3) array of dtype float32 (or anything castable to float32).
        Stored on the instance so that fitness() and with_repr() can be called without re-passing it.
        - repr (list[Triangle], optional): Existing genome. If None, a random one is generated.
        - fitness_metric ({"rmse", "ciede2000"}, default "rmse"): Which distance to optimise. "rmse" is 
        the project spec; "ciede2000" is the Challenge 1 perceptual alternative (see header).
        - target_lab (np.ndarray, optional): Precomputed Lab encoding of ``target``. Passed through ``with_repr``
        so siblings of an individual share the same array (Lab conversion of a 300x400 image is not free and is 
        identical across the population).
        - max_triangle_size (float, default 0.25): Domain constraint: the bounding box of each triangle is limited 
        to this fraction of the canvas size. Passed through to each Triangle so that they can enforce it at 
        construction and mutation time.
        - alpha_min (float, default 0.30): Domain constraint: the alpha gene is clipped to this minimum, so that 
        every triangle contributes at least this much to the rendered image and is not a wasted gene.
        - alpha_max (float, default 0.80): Domain constraint: the alpha gene is clipped to this maximum, so that 
        no triangle is fully opaque and hides the triangles below it, defeating the alpha-blending stacking that 
        gives the rendered painting its smooth gradients.
"""

    def __init__(
        self,
        target: np.ndarray,
        repr=None,
        fitness_metric: str = "rmse",
        target_lab: np.ndarray = None,
        max_triangle_size: float = 1,
        alpha_min: float = 0,
        alpha_max: float = 1,
    ):
        self.target = target
        self.fitness_metric = fitness_metric
        self.max_triangle_size = max_triangle_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        if fitness_metric == "ciede2000" and target_lab is None:
            target_lab = rgb_to_lab(target)

        self.target_lab = target_lab

        if repr is None:
            repr = self.random_initial_representation()

        # Defensive copy of each Triangle so children of crossover cannot
        # share Triangle instances with their parents.
        self.repr = [t.copy() for t in repr]

        self._fitness = None  # populated on first fitness() call





    def random_initial_representation(self):
        """Generate a random genome: a list of N_TRIANGLES Triangles, with the same Domain constraints
        
        Returns:
            - list[Triangle]: A list of N_TRIANGLES Triangle instances, each initialized with the specified
            domain constraints.
        
        """
        return [Triangle(
            max_triangle_size=self.max_triangle_size,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max
        ) for _ in range(N_TRIANGLES)]




    def with_repr(self, new_repr):
        """Build a new Individual that carries the same target image but a
        different genome. Mutation and crossover MUST go through this
        method so that _fitness is left as None on the child and gets
        recomputed from the new genome.

        Parameters:
            - new_repr (list[Triangle]): The genome for the new Individual.
        
        Returns:
            - Individual: A new Individual instance with the same target and fitness metric but the provided genome
        """
        return Individual(
            target=self.target,
            repr=new_repr,
            fitness_metric=self.fitness_metric,
            target_lab=self.target_lab,
            max_triangle_size=self.max_triangle_size,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
    



    def render(self, antialiased=False) -> Image.Image:
        """Rasterize the genome into an RGB PIL image.

        The phenotype is rendered by drawing each triangle onto its own
        transparent RGBA layer and alpha-compositing that layer over a base
        canvas. This preserves correct Porter-Duff 'over' blending, so
        semi-transparent triangles accumulate visually in the intended order.

        When antialiasing is enabled, rendering is performed at a higher
        resolution and then downsampled using Lanczos resampling to reduce
        jagged edges along triangle boundaries.

        Args:
            antialiased (bool): If True, render using supersampling before
                resizing to the target image resolution.

        Returns:
            Image.Image: Rendered RGB phenotype image.
        """

        # Render at higher resolution if supersampling is enabled.
        w = IMG_WIDTH * (SUPERSAMPLE_FACTOR if antialiased else 1)
        h = IMG_HEIGHT * (SUPERSAMPLE_FACTOR if antialiased else 1)

        # Start with a fully opaque black background
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))

        for triangle in self.repr:
            # Each triangle is drawn on its own transparent layer so that
            # alpha blending can be composited correctly onto the canvas
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw  = ImageDraw.Draw(layer)

            if antialiased:
                # Convert normalised coordinates [0,1] into supersampled pixel coordinates.
                g = triangle.repr
                verts = [
                    (round(g[0] * (w - 1)), round(g[1] * (h - 1))),
                    (round(g[2] * (w - 1)), round(g[3] * (h - 1))),
                    (round(g[4] * (w - 1)), round(g[5] * (h - 1))),
                ]
            else:
                verts = triangle.vertices()

            # Rasterise the filled triangle with RGBA colour.
            draw.polygon(verts, fill=triangle.color())

            # Alpha composite triangle layer over the existing canvas.
            canvas = Image.alpha_composite(canvas, layer)

        if antialiased:
            # Downsample back to target resolution using Lanczos filtering to smooth triangle edges.
            canvas = canvas.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

        # Final image returned as RGB (alpha channel discarded).
        return canvas.convert("RGB")







    def fitness(self) -> float:
        """Compute the fitness of the genome relative to the target image.

        Fitness measures how visually close the rendered phenotype is to the
        target image. Lower values indicate a better match.

        Two metrics are supported:

        - ``rmse``:
        Root Mean Square Error computed directly in sRGB space.
        Measures raw pixel-wise numerical difference.

        - ``ciede2000``:
        Mean per-pixel CIEDE2000 ΔE computed in CIE Lab space.
        Measures perceptual colour difference based on human vision.

        Fitness is cached after the first computation and reused until the
        genome is modified.

        Returns:
            - float: Fitness score (lower is better).
        """

        # Return cached value if already computed.
        if self._fitness is not None:
            return self._fitness
        
        # Render phenotype into image array.
        rendered = np.asarray(self.render(), dtype=np.float32)

        if self.fitness_metric == "ciede2000":
            target_lab = self.target_lab if self.target_lab is not None else rgb_to_lab(self.target)
            rendered_lab = rgb_to_lab(rendered)

            # Mean perceptual ΔE across all pixels.
            self._fitness = float(np.mean(ciede2000(rendered_lab, target_lab)))

        else:
            # Standard RMSE in RGB space.
            diff = rendered - self.target.astype(np.float32, copy=False)
            self._fitness = float(np.sqrt(np.mean(diff * diff)))

        return self._fitness
    





    def plot(self, ax=None, title="Solution") -> None:
        """Display the rendered phenotype using matplotlib.

        If an axes object is provided, the image is drawn onto that axes.
        Otherwise a new figure and axes are created automatically.

        Args:
            ax (matplotlib.axes.Axes | None): Optional axes to draw on.
            title (str): Plot title.

        Returns:
            None
        """
        show = ax is None
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(3, 4))

        # Render and display phenotype image.
        ax.imshow(self.render())
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if show:
            plt.tight_layout()
            plt.show()


            

    def __repr__(self) -> str:
        """Return the string representation of the genome.

        Provides a readable representation of the underlying genome data,
        typically the list of triangle genes stored in ``self.repr``.

        Returns:
            str: String representation of the genome.
        """
        return str(self.repr)
