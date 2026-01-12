"""
main initiater
"""
import sys
import asyncio
from pathlib import Path
from detect_text import detect_and_draw_text
import logging
import os

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

root_dir = "t_01"
output_folder = "translations"
files = sorted(os.listdir(root_dir))

def main(verbose:bool):
    level = (logging.DEBUG or logging.WARNING) if verbose else logging.ERROR

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] : %(message)s"
    )

    for file in range(len(files)):
        asyncio.run(detect_and_draw_text(root_dir + "/" + files[file] , output_folder))
        print(f"processed {file + 1} file...")

if __name__ == "__main__":
    main(verbose=False)