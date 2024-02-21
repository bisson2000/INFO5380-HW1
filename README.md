# INFO5380-HW1
INFO5380: Digital Fabrication - HW1


## Cloning the repository

```
git clone git@github.com:bisson2000/INFO5380-HW1.git
```

```
cd INFO5380-HW1
```

## Requirements

Python version >= 3.8

## Getting Started

### Creating a virtual environment

1. From the root repository, create a local virtual environment with `python -m venv venv`
2. Activate the virtual environment with
    1. Windows: `source venv/Scripts/activate`
    2. MacOS/Linux: `source venv/bin/activate`
3. Install the requirement with `pip install requirements.txt`

### Using the program

1. From the root repository, run `python svg2xyz.py`
2. A file explorer should open. Select any SVG, PNG or JPG
3. The program should terminate successfully and create an `output.csv`
4. In the wire terminal, import the `output.csv`
5. In the wire terminal, start printing the shape after importing.

## Useful resources

SVG visualiser:
[https://svg-path-visualizer.netlify.app/](https://svg-path-visualizer.netlify.app/)

SVG guide:
[https://css-tricks.com/svg-path-syntax-illustrated-guide/](https://css-tricks.com/svg-path-syntax-illustrated-guide/)

