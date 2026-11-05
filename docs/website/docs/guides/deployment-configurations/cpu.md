---
hide:
  - tags
tags:
  - CPU
icon: octicons/cpu-16
---

# CPU deployment

IREE supports efficient program execution on CPU devices by using
[LLVM](https://llvm.org/) to compile all dense computations in each program into
highly optimized CPU native instruction streams, which are embedded in one of
IREE's deployable formats.

To compile a program for CPU execution:

1. Pick a CPU target supported by LLVM. By default, IREE includes these LLVM
   targets:

    * X86
    * ARM
    * AArch64
    * RISCV

    Other targets may work, but in-tree test coverage and performance work is
    focused on that list.

2. Pick one of IREE's supported executable formats:

    | Executable Format | Description                                           |
    | ----------------- | ----------------------------------------------------- |
    | Embedded ELF      | (Default) Portable, high performance dynamic library  |
    | System library    | Platform-specific dynamic library (.so, .dll, etc.)   |
    | VMVX              | Reference target                                      |

At runtime, CPU executables can be loaded using one of IREE's CPU HAL devices:

* `local-task`: asynchronous, multithreaded device built on IREE's "task"
   system
* `local-sync`: synchronous, single-threaded devices that executes work inline

## :octicons-download-16: Prerequisites

### Get the IREE compiler

#### :octicons-download-16: Download the compiler from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-compiler`](https://pypi.org/project/iree-base-compiler/)
package includes the compiler tools:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-compiler-from-release.md"

#### :material-hammer-wrench: Build the compiler from source

Please make sure you have followed the
[Getting started](../../building-from-source/getting-started.md) page to build
IREE for your host platform. The `llvm-cpu` compiler backend is compiled in by
default on all platforms, though you should ensure that the
`IREE_TARGET_BACKEND_LLVM_CPU` CMake option is `ON` when configuring.

!!! tip
    `iree-compile` will be built under the `iree-build/tools/` directory. You
    may want to include this path in your system's `PATH` environment variable.

### Get the IREE runtime

You will need to get an IREE runtime that supports the local CPU HAL driver,
along with the appropriate executable loaders for your application.

#### :octicons-download-16: Download the runtime from a release

Python packages are distributed through multiple channels. See the
[Python Bindings](../../reference/bindings/python.md) page for more details.
The core [`iree-base-runtime`](https://pypi.org/project/iree-base-runtime/)
package includes the local CPU HAL drivers:

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-runtime-from-release.md"

#### :material-hammer-wrench: Build the runtime from source

Please make sure you have followed one of the
[Building from source](../../building-from-source/index.md) pages to build
IREE for your target platform. The local CPU HAL drivers and devices are
compiled in by default on all platforms, though you should ensure that the
`IREE_HAL_DRIVER_LOCAL_TASK` and `IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF`
(or other executable loader) CMake options are `ON` when configuring.

#### :octicons-checklist-24: Check for CPU devices

You can check for CPU support by looking for the `local-sync` and `local-task`
drivers and devices:

```shell
iree-run-module --list_drivers
```

```console hl_lines="9-10"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-driver-list.md:2"
```

```shell
iree-run-module --list_devices
```

```console hl_lines="3-4"
--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-run-module-device-list-amd.md"
```

## Compile and run a program

With the requirements out of the way, we can now compile a model and run it.

### :octicons-file-code-16: Compile a program

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-import-onnx-mobilenet.md"

Then run the following command to compile with the `local` device, `llvm-cpu`
target compilation backend, and recommended optimization flags. See
[Choosing CPU targets](#choosing-cpu-targets) for how to select the
instruction set for your CPU:

``` shell hl_lines="2-4"
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-opt-level=O2 \
    --iree-opt-data-tiling \
    mobilenetv2.mlir -o mobilenet_cpu.vmfb
```

--8<-- "docs/website/docs/guides/deployment-configurations/snippets/_iree-optimization-options.md"

???+ tip "Tip - Data-Tiling"

    Use `--iree-opt-data-tiling` to enable the optimization that IREE developers
    have been working on. The option is default off for many reasons, but it has
    been used for performance when users target CPU. See
    [Data-Tiling](../../reference/optimization-options.md#data-tiling-iree-opt-data-tiling-off)
    for more details.

#### Choosing CPU targets

Modern CPUs support optional instruction set extensions, such as AVX2 and
AVX-512 on x86, NEON, SVE and dot-product instructions on Arm, and the vector
extension on RISC-V. IREE only uses these extensions if you tell the compiler
that your target CPU supports them.

!!! warning

    If you don't specify a target CPU, the compiler uses a generic CPU for the
    target architecture. Generic code runs everywhere but leaves out the
    vector and matrix instructions that make ML workloads fast.

    If you specify instructions that your CPU doesn't support, the program
    will compile but will crash at runtime with an "illegal instruction" error.

IREE gives you three flags to describe the target CPU:

| Flag | What it selects | Example values |
| ---- | --------------- | -------------- |
| `--iree-llvmcpu-target-triple` | The architecture and operating system | `x86_64-linux-gnu`, `aarch64-linux-android34`, `riscv64-linux-gnu` |
| `--iree-llvmcpu-target-cpu` | A named CPU model, which implies its full set of instructions | `host`, `znver4`, `sapphirerapids`, `x86-64-v3`, `neoverse-v2`, `apple-m1` |
| `--iree-llvmcpu-target-cpu-features` | Individual instruction set extensions to add or remove | `+avx2,+fma`, `+sve,+i8mm`, `+v` |

Choose the option below that matches your situation.

##### Run on the machine that compiles the program

If you run the compiled program on the same machine that runs
`iree-compile`, pass `host`:

```shell
--iree-llvmcpu-target-cpu=host
```

The compiler detects your CPU model and all of the instruction set extensions
it supports, and generates code that uses them. This is the right choice for
most users. The target triple also defaults to your host,
so you don't need to set it.

##### Run on a different machine of the same architecture

If you compile on one machine and deploy to another, `host` describes the
wrong CPU. Instead, name the CPU of the machine that runs the program. For
example, for a server with an AMD Zen 4 CPU:

```shell
--iree-llvmcpu-target-cpu=znver4
```

The compiler fills in the instruction set extensions that the named CPU
supports. This works for x86, AArch64 and RISC-V 64 targets.

To find the name for a CPU, run the following on the deployment machine if
you have [Clang](https://clang.llvm.org/) or GCC installed:

=== "Clang"

    ```shell
    clang -march=native -### -c -x c /dev/null 2>&1 | grep -o '"-target-cpu" "[^"]*"'
    ```

=== "GCC (x86)"

    ```shell
    gcc -march=native -Q --help=target | grep -- '-march='
    ```

Some common values are:

| Target | CPU names |
| ------ | --------- |
| Intel x86 | `skylake`, `icelake-server`, `sapphirerapids`, `alderlake` |
| AMD x86 | `znver2`, `znver3`, `znver4`, `znver5` |
| Arm servers | `neoverse-n1`, `neoverse-v1`, `neoverse-v2` |
| Apple silicon | `apple-m1`, `apple-m2`, `apple-m3`, `apple-m4` |

!!! tip "Target a family of x86 CPUs"

    To build one program that runs on many x86 machines, use one of the
    x86-64 microarchitecture levels instead of a specific CPU model:

    | Level | Adds |
    | ----- | ---- |
    | `x86-64-v2` | SSE4.2, POPCNT |
    | `x86-64-v3` | AVX2, FMA, BMI2 (most x86 CPUs since 2015) |
    | `x86-64-v4` | AVX-512 |

    For example, `--iree-llvmcpu-target-cpu=x86-64-v3` produces code that runs
    on any CPU with AVX2.

##### Add or remove individual instruction set extensions

Use `--iree-llvmcpu-target-cpu-features` to fine-tune the instruction set.
List the features separated by commas, with `+` to enable a feature or `-` to
disable it. Features you list are combined with those implied by
`--iree-llvmcpu-target-cpu`. For example, to target a Neoverse V1 CPU but
disable SVE:

```shell
--iree-llvmcpu-target-cpu=neoverse-v1 \
--iree-llvmcpu-target-cpu-features=-sve
```

!!! note

    `host` must be used either for both `--iree-llvmcpu-target-cpu` and
    `--iree-llvmcpu-target-cpu-features`, or not at all. Combining `host` with
    an explicit CPU name or feature list is an error.

##### Cross-compile for a different architecture

To compile on one architecture and run on another, for example on an x86
workstation for an Arm phone, also set `--iree-llvmcpu-target-triple`:

```shell
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=aarch64-linux-android34 \
    --iree-llvmcpu-target-cpu=cortex-a715 \
    --iree-opt-level=O2 \
    --iree-opt-data-tiling \
    mobilenetv2.mlir -o mobilenet_android.vmfb
```

To list the architectures that your copy of `iree-compile` can generate code
for, run:

```shell
iree-compile --iree-llvmcpu-list-targets
```

```text
  Registered Targets:
    aarch64    - AArch64 (little endian)
    aarch64_32 - AArch64 (little endian ILP32)
    aarch64_be - AArch64 (big endian)
    arm        - ARM
    arm64      - ARM64 (little endian)
    arm64_32   - ARM64 (little endian ILP32)
    armeb      - ARM (big endian)
    riscv32    - 32-bit RISC-V
    riscv64    - 64-bit RISC-V
    wasm32     - WebAssembly 32-bit
    wasm64     - WebAssembly 64-bit
    x86        - 32-bit X86: Pentium-Pro and above
    x86-64     - 64-bit X86: EM64T and AMD64
```

See the [bare-metal guide](./bare-metal.md) for
targets without an operating system.

### :octicons-terminal-16: Run a compiled program

To run the compiled program:

``` shell hl_lines="2"
iree-run-module \
    --device=local-task \
    --module=mobilenet_cpu.vmfb \
    --function=torch-jit-export \
    --input="1x3x224x224xf32=0"
```

The above assumes the exported function in the model is named `torch-jit-export`
and it expects one 224x224 RGB image. We are feeding in an image with all 0
values here for brevity, see `iree-run-module --help` for the format to specify
concrete values.

<!-- TODO(??): measuring performance -->

<!-- TODO(??): troubleshooting -->
