import torch
import deepspeed
import subprocess
from .ops.op_builder import ALL_OPS
from .git_version_info import installed_ops
from .ops import __compatible_ops__ as compatible_ops

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
END = '\033[0m'
SUCCESS = f"{GREEN} [SUCCESS] {END}"
OKAY = f"{GREEN}[OKAY]{END}"
WARNING = f"{YELLOW}[WARNING]{END}"
FAIL = f'{RED}[FAIL]{END}'
INFO = '[INFO]'

color_len = len(GREEN) + len(END)
okay = f"{GREEN}[OKAY]{END}"
warning = f"{YELLOW}[WARNING]{END}"


def op_report():
    max_dots = 23
    max_dots2 = 11
    h = ["op name", "installed", "compatible"]
    print("DeepSpeed cpp/cuda extension op report")
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print("JIT compiled ops requires ninja")
    ninja_status = OKAY if ninja_installed() else FAIL
    print('ninja', "." * (max_dots - 5), ninja_status)
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print(h[0], "." * (max_dots - len(h[0])), h[1], "." * (max_dots2 - len(h[1])), h[2])
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    installed = f"{GREEN}[YES]{END}"
    no = f"{YELLOW}[NO]{END}"
    for op_name, builder in ALL_OPS.items():
        dots = "." * (max_dots - len(op_name))
        is_compatible = OKAY if builder.is_compatible() else no
        is_installed = installed if installed_ops[op_name] else no
        dots2 = '.' * ((len(h[1]) + (max_dots2 - len(h[1]))) -
                       (len(is_installed) - color_len))
        print(op_name, dots, is_installed, dots2, is_compatible)
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))


def ninja_installed():
    result = subprocess.Popen('type ninja', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def debug_report():
    max_dots = 23
    report = [
        ("torch install path",
         torch.__path__),
        ("torch version",
         torch.__version__),
        ("torch cuda version",
         torch.version.cuda),
        ("deepspeed install path",
         deepspeed.__path__),
        ("deepspeed info",
         f"{deepspeed.__version__}, {deepspeed.__git_hash__}, {deepspeed.__git_branch__}"
         )
    ]
    print("DeepSpeed general environment info:")
    for name, value in report:
        print(name, "." * (max_dots - len(name)), value)


def main():
    op_report()
    debug_report()


if __name__ == "__main__":
    main()