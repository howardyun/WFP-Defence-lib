from __future__ import annotations

import platform
import sys

from _bootstrap import ensure_project_root


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    try:
        import torch
    except ImportError:
        print("torch: not installed")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cudnn_available: {torch.backends.cudnn.is_available()}")
    print(f"cudnn_version: {torch.backends.cudnn.version()}")

    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gib = props.total_memory / (1024 ** 3)
            print(f"cuda_device_{index}: {props.name}, capability={props.major}.{props.minor}, memory={total_gib:.2f} GiB")

    for package_name in ("numpy", "tqdm"):
        try:
            package = __import__(package_name)
        except ImportError:
            print(f"{package_name}: not installed")
        else:
            print(f"{package_name}: {getattr(package, '__version__', 'unknown')}")

    project_root = ensure_project_root(required_modules=("cluster",))
    import toob
    import toob.cluster

    print(f"project_root: {project_root}")
    print(f"toob: {toob.__file__}")
    print(f"toob.cluster: {toob.cluster.__file__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
