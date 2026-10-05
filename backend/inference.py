"""Convenience wrapper for the LesionIQ inference CLI.

The implementation lives in backend/classifier/inference.py so imports remain
compatible with the backend package layout.
"""

from classifier.inference import *  # noqa: F401,F403
from classifier.inference import main


if __name__ == "__main__":
    main()
