# QR Stitcher & Solver

A collection of OpenCV-based utilities for reconstructing and decoding QR codes from fragments or physical puzzles.

## Tools

- **`qr_stitcher.py`**: A manual placement tool where you can click on a "puzzle" image to place a QR fragment and attempt to decode the URL.
- **`qr_auto_stitch.py`**: An attempt at automated fragment reconstruction using feature matching.
- **`qr_manual.py`**: Manual coordinate-based stitching for high-precision reconstruction.

## Examples

The repository includes several sample images used for testing:
- `PXL_20251223_005656289.jpg`: The base image with a missing fragment.
- `Screenshot_20251222-175703.png`: The isolated fragment.
- `final_solution.png`: The successful reconstruction.

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- `pyzbar` (for QR decoding)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
