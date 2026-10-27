"""
cloud_renderer: standalone CLI that renders a heterogeneous participating
medium (a density volume loaded from a .npy file) under a directional light
and/or an HDR environment.

Uses rdv's next-event-estimation volume renderers (PBVR_NEE for the
directional light, PBVR_ENV - importance-sampled via XREnvironmentSampler,
or UniformEnvironmentSampler for a constant ambient color - for the
environment) rather than plain unidirectional path tracing (PBVRT): NEE
explicitly samples the light/environment at every scattering event instead
of relying on a random walk happening to hit them, which converges to a
clean image in far fewer samples for this kind of scene (point/small-
solid-angle-dominated lighting). Combined as light_intensity*direct_nee +
environment_nee + transmittance*background (transmittance and background
multiplied *per sample*, inside the same dispatch, before averaging - not
averaged separately and multiplied afterward, which would be wrong for two
correlated noisy per-sample estimators: E[T]*E[B] != E[T*B] in general).
Matches the same formula rdv_nr's own ScatNet.render_full_radiance
documents (direct_neural_rendering/src/rdv_nr/_modeling.py) - just without
that class's separate post-hoc environment_scale knob, since this script's
own --light_intensity is the only post-render scale exposed, and the HDR
file's/constant color's own values are otherwise used as-is.

Volume file: single .npy, shape (D, H, W, 1), float - raw extinction/density
grid (scaled at render time by --density_scale).

Environment: either --environment_path (.npy, shape (H, W, 3), float,
equirectangular HDR radiance map) or --environment_color (a constant
ambient RGB color, uniformly sampled - see UniformEnvironmentSampler) -
mutually exclusive. If neither is given, the scene has no visible
background/environment illumination - only the directional light
(--light_direction / --light_intensity) contributes.

Output: if --output_path ends in .npy, saves the raw linear radiance
(H, W, 3) float array. If it ends in .png, applies gamma correction
(x ** (1/2.2), clipped to [0,1]) and saves an 8-bit image.
"""
import argparse
import os
import numpy as np
import torch

import rdv


class UniformEnvironmentSampler(rdv.Map):
    """Uniform-sphere-direction counterpart to rdv.XREnvironmentSampler,
    for a constant ambient color instead of an image environment - same
    Map[6, 3+SPECTRAL_DIM+1] interface (direction | color-weighted-by-
    1/pdf | pdf), with pdf=1/(4*pi) over the whole sphere, so the sampled
    color comes out scaled by 4*pi (weight = 1/pdf). See
    uniform_environment_sampler.h for the shader itself."""
    __code__ = "uniform_environment_sampler.h"
    __requires__ = [rdv.RandomLibrary, rdv.MathLibrary]
    color: rdv.vec3

    def __init__(self, color) -> None:
        self.match(rdv.Generics.IN, 6)
        self.match(rdv.Generics.OUT, 3 + 3 + 1)  # direction(3) + weighted color(3) + pdf(1)
        self.color = rdv.vec3(*color) if not isinstance(color, rdv.vec3) else color


def load_volume(path: str) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 1:
        raise ValueError(f"--volume_path must be a (D,H,W,1) array, got shape {arr.shape}")
    return torch.from_numpy(arr)


def load_environment(path: str | None) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, 1, 3)  # no environment: black background, no env illumination
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"--environment_path must be an (H,W,3) array, got shape {arr.shape}")
    # rdv's own equirectangular v-coordinate convention puts "top" (zenith)
    # at row -1, the opposite of a typical top-row-first image - flip
    # vertically so a normal-convention HDR ends up right-side-up here too.
    arr = np.flip(arr, axis=0).copy()
    return torch.from_numpy(arr)


def build_extinction_map(extinction_tensor: torch.Tensor):
    extinction_tensor = rdv.ensure_on_rdv(extinction_tensor, location=rdv.MemoryLocation.DEVICE)
    extinction_gpu_sampler = rdv.GPUSampler(extinction_tensor, clamp=True, aligned=False, linear=True)
    extinction_map = rdv.RefMap(extinction_gpu_sampler)
    return extinction_tensor, extinction_map


def build_majorant_map(extinction_tensor: torch.Tensor, density_scale: float):
    # same sphere-based majorant construction ScatNet.build_maps() uses
    # (direct_neural_rendering/src/rdv_nr/_modeling.py) - cheap DDA bound
    # for the delta-tracking inside PBVR_NEE/PBVR_ENV.
    div = max(1, extinction_tensor.shape[1] // 32)
    majorant_grid_shape = [max(1, d // div) for d in extinction_tensor.shape[:3]]
    majorant_grid = rdv.TensorGrid(
        rdv.MajorantGrid(rdv.TensorGrid(extinction_tensor), majorant_grid_shape, aligned=False, linear=True)())
    sphere_radius_field_grid = rdv.sphere_radius_field(majorant_grid, levels=16, density_scale=density_scale)
    return rdv.RefMap(rdv.SphereMajorantField(sphere_radius_field_grid))


def build_transform(extinction_tensor: torch.Tensor):
    # fit the volume's own (possibly non-cubic) voxel grid into a
    # consistent unit-ish box, same convention as ScatNet.set_extinction's
    # own new-shape branch.
    bmin, bmax = rdv.grid3d_fit_box(extinction_tensor.shape[:3], align_corners=False)
    scale, offset = rdv.unit2box(bmin, bmax)
    return rdv.trs(translation=offset, scale=scale)


def build_environment_map(environment_tensor: torch.Tensor):
    environment_tensor = rdv.ensure_on_rdv(environment_tensor, location=rdv.MemoryLocation.DEVICE)
    environment_gpu_sampler = rdv.GPUSampler(environment_tensor, clamp=True, aligned=False, linear=True)
    environment_map = rdv.RefMap(rdv.dir2xr >> environment_gpu_sampler)
    environment_sampler = rdv.XREnvironmentSampler(environment_map, environment_tensor, levels=8)
    return environment_map, environment_sampler


def build_environment(args):
    """Returns (environment_map, environment_sampler) for either an HDR
    image (--environment_path) or a constant ambient color
    (--environment_color, mutually exclusive with --environment_path) -
    the latter skips XREnvironmentSampler's image-based importance
    sampling entirely in favor of UniformEnvironmentSampler (every
    direction equally likely, pdf=1/4pi, see that class's own docstring),
    since there's no spatial structure in a constant color to importance-
    sample in the first place."""
    if args.environment_color is not None:
        if args.environment_path is not None:
            raise ValueError("--environment_color and --environment_path are mutually exclusive")
        color = tuple(args.environment_color)
        environment_map = rdv.Constant(rdv.wrap(color).as_tensor('torch'))
        environment_sampler = UniformEnvironmentSampler(color)
        return environment_map, environment_sampler
    return build_environment_map(load_environment(args.environment_path))


def build_camera(args):
    pose = torch.zeros(1, 9)
    pose[0, 0:3] = torch.tensor(args.camera_origin)
    pose[0, 3:6] = torch.tensor(args.camera_target)
    pose[0, 6:9] = torch.tensor(args.camera_up)
    return rdv.Cameras(
        args.width, args.height, poses=pose,
        aspect_ratio=args.width / args.height, fov=args.fov, jitter=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume_path", required=True, help=".npy file, shape (D,H,W,1), raw extinction/density grid")
    parser.add_argument("--environment_path", default=None, help=".npy file, shape (H,W,3), equirectangular HDR radiance (optional)")
    parser.add_argument("--environment_color", type=float, nargs=3, default=None,
                         help="constant ambient color instead of an HDR image - mutually exclusive with --environment_path")
    parser.add_argument("--density_scale", type=float, default=1.0)
    parser.add_argument("--light_direction", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--light_intensity", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                         help="directional light radiance; (0,0,0) disables it (environment-only lighting)")
    parser.add_argument("--scattering_albedo", type=float, nargs=3, default=[.99, .985, .98])
    parser.add_argument("--phase_g", type=float, default=0.0, help="Henyey-Greenstein anisotropy, [-1,1]")
    parser.add_argument("--bounces", type=int, default=32, help="max scattering bounces per path")
    parser.add_argument("--samples", type=int, default=64, help="Monte Carlo samples averaged per pixel")
    parser.add_argument("--batch_samples", type=int, default=16,
                         help="samples evaluated per single GPU dispatch, via average() (lower this if you hit "
                              "ErrorDeviceLost at high resolution/sample counts - that's Windows' TDR watchdog "
                              "killing a dispatch that ran too long, not a real crash)")
    parser.add_argument("--camera_origin", type=float, nargs=3, default=[0.0, 0.0, 3.0])
    parser.add_argument("--camera_target", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--camera_up", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov", type=float, default=45.0, help="vertical field of view, degrees")
    parser.add_argument("--output_path", required=True, help="output file - .npy (raw linear) or .png (gamma-corrected)")
    args = parser.parse_args()

    extinction_tensor, extinction_map = build_extinction_map(load_volume(args.volume_path))
    majorant_map = build_majorant_map(extinction_tensor, args.density_scale)
    transform = build_transform(extinction_tensor)
    environment_map, environment_sampler = build_environment(args)
    scattering_albedo_map = rdv.Constant(rdv.wrap(tuple(args.scattering_albedo)).as_tensor('torch'))
    sensor = build_camera(args)

    direct_nee_map = rdv.PBVR_NEE(
        extinction=extinction_map, majorant=majorant_map, scattering_albedo=scattering_albedo_map,
        transform=transform, light_direction=tuple(args.light_direction),
        density_scale=args.density_scale, g=args.phase_g, max_bounces=args.bounces,
    )
    env_nee_map = rdv.PBVR_ENV(
        extinction=extinction_map, majorant=majorant_map, scattering_albedo=scattering_albedo_map,
        environment_sampler=environment_sampler, transform=transform,
        density_scale=args.density_scale, g=args.phase_g, max_bounces=args.bounces,
    )
    transmittance_map = rdv.RatiotrackingTransmittance(
        extinction=extinction_map, majorant=majorant_map, transform=transform, density_scale=args.density_scale,
    )
    # single combined dispatch: direct-light NEE (3ch) | environment NEE
    # (3ch) | transmittance (1ch) | background env along the ray's own
    # direction (3ch) - same multi-map-concat-then-slice pattern used
    # throughout rdv_nr's own ScatNet subclasses.
    combined_map = direct_nee_map | env_nee_map | (transmittance_map * (rdv.X[3:6] >> environment_map))
    # average(n) evaluates all n samples inside a single GPU dispatch (see
    # its own docstring) - at high resolution/sample counts that single
    # dispatch can run long enough to trip Windows' TDR watchdog (driver
    # reset, ErrorDeviceLost), so split into several smaller dispatches
    # accumulated on-device instead, each small enough to finish quickly.
    chunk = max(1, args.batch_samples)
    capture_grid = sensor >> combined_map >> rdv.average(chunk)

    print(f"Rendering {args.width}x{args.height}, {args.samples} samples ({chunk}/dispatch), "
          f"{args.bounces} max bounces (PBVR_NEE direct + PBVR_ENV environment)...")
    # everything stays on the GPU through accumulation - only the final
    # result is moved to host memory, right before converting to numpy.
    accum = torch.zeros(args.height, args.width, 9, device='cuda')
    done = 0
    while done < args.samples:
        n = min(chunk, args.samples - done)
        c = capture_grid().as_tensor('torch')[0]
        accum += c * n
        done += n
        print(f"  {done}/{args.samples} samples")
    c = accum / args.samples
    light_intensity = torch.tensor(args.light_intensity, device='cuda')
    direct, env_scatter, background = c[..., 0:3], c[..., 3:6], c[..., 6:9]
    result = direct * light_intensity + env_scatter + background
    # rdv's own renderer produces images with row 0 = bottom (matching the
    # environment's own v-coordinate convention - see load_environment) -
    # flip back to normal top-row-first order before saving.
    image = torch.flip(result, dims=[0]).cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    ext = os.path.splitext(args.output_path)[1].lower()
    if ext == ".npy":
        np.save(args.output_path, image)
    elif ext == ".png":
        from PIL import Image
        srgb = np.clip(image, 0.0, 1.0) ** (1.0 / 2.2)
        Image.fromarray((srgb * 255).astype(np.uint8)).save(args.output_path)
    else:
        raise ValueError(f"--output_path must end in .npy or .png, got {args.output_path!r}")
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
