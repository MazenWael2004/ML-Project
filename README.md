# Project Requirements and Structure

## Dependencies
To run the notebook (`project.ipynb`), you need to install the following Python packages:

- `tqdm`
- `opencv-python`
- `torch`
- `torchvision`
- `Pillow`

You can install them using pip:

```bash
pip install tqdm opencv-python torch torchvision Pillow
```

## Project Structure
- The `dataset` directory **must be located in the project root directory** and should be named exactly `dataset`.
- The Jupyter notebook file (`project.ipynb`) should also be in the project root directory.

Example structure:

```
ML-Project/
├── project.ipynb
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── ...
```
