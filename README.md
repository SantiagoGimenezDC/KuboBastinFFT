# KuboBastinFFT
FFT solver for the Kubo Bastin formula

## Building

Requires a C++17 compiler with OpenMP, Eigen 3, FFTW 3 and Boost (math headers).

```bash
make standard
```

## Running

```bash
OMP_NUM_THREADS=8 ./KuboBastinFFT input.toml
```

The input is a [TOML](https://toml.io) file. See `examples/template/input.toml` for a documented
example listing every option. Keys the chosen device or solver does not use are reported as errors,
so typos are caught before the run starts. A copy of the input file is stored in the output directory.

Old positional input files (any extension other than `.toml`) are still read, but that format is deprecated.
