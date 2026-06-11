<p align="center">
   <img src="https://raw.githubusercontent.com/idia-pipelines/idia-pipelines.github.io/master/assets/idia_logo.jpg" alt="IDIA pipelines"/>
</p>

# The IDIA MeerKAT Pipeline (v2.0)

The IDIA MeerKAT pipeline is a radio interferometric calibration pipeline designed to process MeerKAT data. It implements cross-calibration, self-calibration, and science imaging.

## Requirements

- **Python**: >= 3.10
- **CASA**: Required for all processing (usually provided via a container).
- **Cluster**: Designed for SLURM-based clusters (e.g., Ilifu).
- **Dependencies**: `tomli` (for Python < 3.11).

## Installation

The pipeline is now packaged as a professional Python library.

### Using pip
Clone the repository and install it into your environment:
```bash
git clone https://github.com/idia-astro/pipelines.git
cd IDIA_pipeline
pip install .
```
For developers who wish to modify the code, use an editable install:
```bash
pip install -e .
```

## Quick Start

**Note: It is not necessary to copy the raw data (i.e. the MS) to your working directory. The pipeline handles MS/MMS creation and does not manipulate raw data stored in read-only projects directories.**

### 1. Build a config file
The pipeline uses **TOML** for configuration, providing a human-readable and structured way to define the processing flow.

#### a. For continuum/spectral line processing:
```bash
processMeerKAT -B -C myconfig.toml -M mydata.ms
```

#### b. For polarization processing:
```bash
processMeerKAT -B -C myconfig.toml -M mydata.ms -P
```

#### c. Including self-calibration:
```bash
processMeerKAT -B -C myconfig.toml -M mydata.ms -2
```

#### d. Including science imaging:
```bash
processMeerKAT -B -C myconfig.toml -M mydata.ms -I
```

This generates a `myconfig.toml` file. You can edit this file to adjust SLURM resources (`nodes`, `mem`, `partition`) or modify the sequence of scripts.

### 2. Run the pipeline
```bash
processMeerKAT -R -C myconfig.toml
```
This will create `submit_pipeline.sh`, which you can then run to submit all pipeline jobs to the SLURM queue:
```bash
./submit_pipeline.sh
```

### Monitoring and Maintenance
- `summary.sh`: Provides a brief overview of job status.
- `findErrors.sh`: Checks log files for commonly reported errors.
- `killJobs.sh`: Kills all jobs from the current pipeline run.
- `cleanup.sh`: Wipes all intermediate data products.

For a full list of command line options, run `processMeerKAT -h`.

---

## 🚀 New Features in v2.0

### 🔌 Script Plugin API
You can now contribute or use custom scripts without modifying the pipeline core.
1. Place your Python script (e.g., `my_custom_flag.py`) in your current working directory.
2. Add the script to the `scripts` array in your `config.toml`:
   ```toml
   scripts = [
     { script = "my_custom_flag.py", mpi = true },
     { script = "setjy.py", mpi = true },
     # ...
   ]
   ```
The pipeline automatically discovers scripts in the current directory first, allowing you to override built-in scripts with local versions.

### 🐳 Container & Runner Support
The pipeline now supports an explicit execution runner, making it easy to use Singularity, Apptainer, or Conda environments.
```bash
processMeerKAT -B -C config.toml -M data.ms --runner "singularity exec /path/to/casa.sif"
```
The `--runner` prefix is embedded into every SLURM script and is used during the `-B` phase for MS metadata extraction.

### 🏢 Facility Abstraction
The pipeline is no longer locked to a single cluster. It uses a facility abstraction layer that allows it to run on any SLURM-based facility. Facility-specific limits (memory, node counts) are handled via `FacilityConfig` profiles.

### 📊 Runtime State Management
To ensure your original `config.toml` remains a template, the pipeline creates a `pipeline_state.toml` at runtime. 
- **`config.toml`**: User-defined settings (read-only during execution).
- **`pipeline_state.toml`**: Runtime state (tracks the current working MS, calibration tables, and progress).

## Using multiple spectral windows (SPW Splitting)

The pipeline splits the MeerKAT band into several spectral windows (SPWs) and processes each concurrently to maximize cluster throughput.

1. **Calibration output**: All output specific to an SPW (calibration tables, logs, plots) is stored within that SPW's dedicated directory.
2. **Top-level logs**: Logs in the root directory correspond to `precal_scripts` and `postcal_scripts` (e.g., `partition.py` and `concat_caltables.py`).

For more detailed information, refer to the documentation on the [pipelines website](https://idia-pipelines.github.io/docs/processMeerKAT).
