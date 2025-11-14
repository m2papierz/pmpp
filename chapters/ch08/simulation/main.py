from dataclasses import dataclass
from pathlib import Path

from .allen_cahn_solver import AllenCahnSolver


@dataclass
class Config:
    field_min: float = -0.5
    field_max: float = 0.5
    dim: int = 256
    n_steps: int = 800
    dh: float = 0.75  # grid spacing
    dt: float = 0.02  # time step
    eps: float = 1.0  # epsilon in Allen-Cahn
    save_every: int = 5  # animation stapes
    anim_fps: int = 20  # fps of the animation


def main():
    lib_path = Path(__file__).parent.parent / "build/libdiffusion_cuda.so"
    output_path = Path(__file__).parent / "allen_cahn.gif"

    with AllenCahnSolver(
        library_path=lib_path,
        field_min=Config.field_min,
        field_max=Config.field_max,
        dim=Config.dim,
        n_steps=Config.n_steps,
        dh=Config.dh,
        dt=Config.dt,
        eps=Config.eps,
    ) as solver:
        solver.save_mid_slice_animation(
            output_path=output_path,
            n_steps=Config.n_steps,
            save_every=Config.save_every,
            fps=Config.anim_fps,
        )


if __name__ == "__main__":
    main()
