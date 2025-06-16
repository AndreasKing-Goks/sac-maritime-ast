# SAC-Maritime-AST

> ⚠️ **Project Deprecated / Unmaintained**
>
> This repository is no longer maintained as we plan to redesign the entire architecture from scratch.  
> The current implementation is outdated and will be replaced by a new version in a separate repository.
>
> **No support or further updates will be provided in this repository.**


Repository for **Soft Actor-Critic Adaptive Stress Testing (SAC-AST)** framework in maritime applications.

---

## Environment Setup

### TensorFlow-GPU Compatibility

Before installation, ensure compatibility between **TensorFlow**, **CUDA**, and **cuDNN**:

- [TensorFlow GPU compatibility guide](https://www.tensorflow.org/install/source#gpu)
- [Windows native pip install instructions](https://www.tensorflow.org/install/pip#windows-wsl2)

> **Note:** This project runs on a local Windows-`WSL2` machine.

### TensorFlow Dependencies

~~All required packages are listed in [`sac-test-gpu.yml`](sac-test-gpu.yml).~~

- ~~Downgrade `numpy` to **<2.0** due to known compatibility issues with TensorFlow.~~

To verify `tensorflow` with GPU support installation:

```bash
python test_beds/test_gpu.py
```

## Installing `rllab` Package

To run the SAC implementation, install the `rllab` dependency from Haarnoja’s original SAC repository:  
[https://github.com/haarnoja/sac/tree/master](https://github.com/haarnoja/sac/tree/master)

---

### Updating PYTHONPATH (IMPORTANT!)

To ensure Python can import both `sac-maritime-ast` codebase and the internal `rllab` module, update the `PYTHONPATH` inside the `~/.bashrc` of the `WSL2` bash terminal.

First open the `~/.bashrc` file.
```bash
nano ~/.bashrc
```

Make sure you are in the project root, then add this line at the end of the file.
```bash
export PYTHONPATH="$(pwd):$(pwd)/rllab:$PYTHONPATH"
```

Save the file. Then source the file.
```bash
source ~/.bashrc
```


### For Visual Studio Code Users

Ensure VS Code recognizes the local `rllab` package by updating your workspace settings.

Edit or create the file `.vscode/settings.json` in your project root and add the following:

```json
{
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "D:/OneDrive - NTNU/PhD/PhD_Projects/sac-maritime-ast/rllab"
  },
  "python.analysis.extraPaths": [
    "rllab"
  ]
}
```

To verify `rllab` package installation:

```bash
python test_beds/test_rllab.py
```
