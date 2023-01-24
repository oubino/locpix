#!/usr/bin/env python
"""Get file names without extensions from
"""
import os
import argparse


def main():

    parser = argparse.ArgumentParser(description="Get file names")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--extension",
        action="store",
        type=str,
        help="extension to remove",
        required=True,
    )

    args = parser.parse_args()

    folder = os.path.join(args.project_directory, "preprocess/no_gt_label")
    files = os.listdir(folder)

    for file in files:
        print(file.removesuffix(args.extension))


if __name__ == "__main__":
    main()
