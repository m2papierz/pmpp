import ctypes
from logging import getLogger
from pathlib import Path

import numpy as np

logger = getLogger(__file__)

c_float_p = ctypes.POINTER(ctypes.c_float)


def load_cuda_library(library_path: Path):
    """
    Load CUDA shared library and bind the C function interfaces.

    Parameters
    ----------
    library_path : Path
        Path to the compiled `libdiffusion_cuda.so` shared library.

    Returns
    -------
    ctypes.CDLL
        Loaded CUDA library handle.
    """
    if not library_path.exists():
        raise FileNotFoundError(f"Shared library not found: {library_path}")

    try:
        lib = ctypes.cdll.LoadLibrary(str(library_path))
    except OSError as exc:
        raise RuntimeError(f"Failed to load CUDA library '{library_path}'") from exc

    lib.allenCahnStep.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input field
        ctypes.c_int,  # dim
        ctypes.c_float,  # dh
        ctypes.c_float,  # dt
        ctypes.c_float,  # eps
    ]
    lib.allenCahnStep.restype = None

    if hasattr(lib, "cleanupCuda"):
        lib.cleanupCuda.argtypes = []
        lib.cleanupCuda.restype = None

    logger.info("Succesfully loaded CUDA library")
    return lib


class AllenCahnSolver:
    """
    Allen-Cahn equation solver using a CUDA backend.

    The solver evolves a scalar field `u(x, y, z, t)` according to

    .. math::
        \\partial_t u = \\epsilon^2 \\nabla^2 u + u - u^3

    on a regular 3D grid using an explicit Euler scheme, with the spatial
    Laplacian and time step implemented in a CUDA shared library.

    Parameters
    ----------
    library_path : Path
        Path to the compiled CUDA shared library.
    field_min : float
        Minimum value for the random initial condition.
    field_max : float
        Maximum value for the random initial condition.
    dim : int
        Linear grid size (the field has shape ``(dim, dim, dim)``).
    n_steps : int
        Default number of time steps to run in :meth:`run`.
    dh : float
        Grid spacing in each spatial direction.
    dt : float
        Time-step size.
    eps : float
        Parameter :math:`\\epsilon` in the Allen-Cahn equation.
    """

    def __init__(
        self,
        library_path: Path,
        field_min: float,
        field_max: float,
        dim: int,
        n_steps: int,
        dh: float,
        dt: float,
        eps: float,
    ):
        self._cuda_lib = load_cuda_library(library_path)

        self.field_min = field_min
        self.field_max = field_max
        self.dim = dim
        self.n_steps = n_steps
        self.dh = dh
        self.dt = dt
        self.eps = eps

        self.u = self.generate_initial_conditions()
        self.current_step: int = 0
        self.current_time: float = 0.0

    def generate_initial_conditions(self) -> np.ndarray:
        """
        Generate the initial condition for the field.

        Returns
        -------
        u0 : ndarray of shape (dim, dim, dim), dtype float32
            Random initial condition sampled uniformly from
            ``[field_min, field_max]``.
        """
        return np.random.uniform(
            self.field_min,
            self.field_max,
            size=(self.dim, self.dim, self.dim),
        ).astype(np.float32)

    def step(self) -> None:
        """
        Perform a single time step of the Allen-Cahn dynamics.

        Updates
        -------
        u : ndarray
            Field is updated in place.
        current_step : int
            Incremented by 1.
        current_time : float
            Increased by ``dt``.
        """
        # Ensure contiguous memory for ctypes
        u = np.ascontiguousarray(self.u, dtype=np.float32)

        self._cuda_lib.allenCahnStep(
            u.ctypes.data_as(c_float_p),
            self.dim,
            self.dh,
            self.dt,
            self.eps,
        )

        self.u = u
        self.current_step += 1
        self.current_time += self.dt

    def run(self, n_steps: int | None = None) -> None:
        """
        Run the simulation for a number of time steps.

        Parameters
        ----------
        n_steps : int, optional
            Number of time steps to perform. If None, uses ``self.n_steps``.
        """
        if n_steps is None:
            n_steps = self.n_steps
        for _ in range(n_steps):
            self.step()

    def save_mid_slice_animation(
        self,
        output_path: Path,
        n_steps: int | None = None,
        save_every: int = 1,
        fps: int = 10,
        cmap: str = "RdBu",
    ) -> None:
        """
        Save an animation of a central slice of the field as a GIF.

        Parameters
        ----------
        output_path : Path
            File path for the output animation (e.g. ``'allen_cahn.gif'``).
        n_steps : int, optional
            Number of steps to simulate for the animation. If None, uses
            ``self.n_steps``.
        save_every : int, default 1
            Save a frame every ``save_every`` steps.
        fps : int, default 10
            Frames per second of the output animation.
        cmap : str, default 'RdBu'
            Matplotlib colormap name for visualization.
        """
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        if n_steps is None:
            n_steps = self.n_steps

        mid = self.dim // 2
        frames = []

        for k in range(n_steps):
            self.step()
            if k % save_every != 0:
                continue

            fig, ax = plt.subplots()
            im = ax.imshow(
                self.u[mid],
                cmap=cmap,
                origin="lower",
                vmin=self.field_min,
                vmax=self.field_max,
            )
            ax.set_title(f"t = {self.current_time:.3f}, step = {self.current_step}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.canvas.draw()

            # Convert canvas to an RGB image array
            w, h = fig.canvas.get_width_height()
            raw = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = raw.reshape(h, w, 4)
            frames.append(buf[..., 1:])
            plt.close(fig)
        if not frames:
            logger.warning("No frames captured, animation not saved.")
            return

        imageio.mimsave(output_path, frames, fps=fps)
        logger.info("Saved animation to %s", output_path)

    def reset(self) -> None:
        """
        Reset the simulation state.

        Resets the field to a new random initial condition and sets
        ``current_step`` and ``current_time`` to zero.
        """
        self.u = self.generate_initial_conditions()
        self.current_step = 0
        self.current_time = 0.0

    def close(self) -> None:
        """
        Release resources associated with the CUDA library, if necessary.
        """
        cleanup = getattr(self._cuda_lib, "cleanupCuda", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error during CUDA cleanup: %s", exc)

    def __enter__(self) -> "AllenCahnSolver":
        """Enter context manager, returning the solver itself."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, ensuring cleanup is called."""
        self.close()

    def __del__(self) -> None:
        # Best effort cleanup; avoid raising in destructor
        try:
            self.close()
        except Exception:
            pass
