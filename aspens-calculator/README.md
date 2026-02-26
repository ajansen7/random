# Aspen's Calculator

A computer vision toolset for digitizing and analyzing board game states, specifically designed for games with hexagonal grids and color-coded components (like "Pine" and "Aspen" trees).

## Tools

- **`aspens-calculator.py`**: A hex grid digitizer. Allows you to trace the perimeter of a board and infer a hexagonal grid. Includes a GUI editor to fine-tune point locations.
- **`tile-finder.py`**: An automated detection script that uses image processing (Top-Hat filters, CLAHE, and blob analysis) to identify tiles and classify them based on color (Pine vs. Aspen).
- **`training.py` / `training-v2.py`**: Scripts for collecting color samples and training the detection models.
- **`calc_v*.py`**: Various iterations of the calculation logic.

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

## Usage

1. Place an image of the board named `board.jpg` in the project root.
2. Run `aspens-calculator.py` to digitize the grid.
3. Run `tile-finder.py` to detect and classify components on the board.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
