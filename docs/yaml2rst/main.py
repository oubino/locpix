#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yaml2rst – A Simple Tool for Documenting YML Files
"""
#
# Copyright 2015-2019 by Hartmut Goebel <h.goebel@crazy-compilers.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function
import re
import argparse
import sys
import os

__author__ = "Hartmut Goebel <h.goebel@crazy-compilers.com>"
__copyright__ = (
    "Copyright 2015-2019 by Hartmut Goebel " "<h.goebel@crazy-compilers.com>"
)
__licence__ = "GNU General Public License version 3 (GPL v3)"
__version__ = "0.3"

STATE_TEXT = 0
STATE_YAML = 1


def setup_patterns():

    # Create rst patterns, copied from `docutils/parsers/rst/states.py`
    # Function has been placed in the public domain.

    class Struct:
        """Stores data attributes for dotted-attribute access."""

        def __init__(self, **keywordargs):
            self.__dict__.update(keywordargs)

    enum = Struct()
    enum.formatinfo = {
        "parens": Struct(prefix="(", suffix=")", start=1, end=-1),
        "rparen": Struct(prefix="", suffix=")", start=0, end=-1),
        "period": Struct(prefix="", suffix=".", start=0, end=-1),
    }
    enum.formats = enum.formatinfo.keys()
    enum.sequences = [
        "arabic",
        "loweralpha",
        "upperalpha",
        "lowerroman",
        "upperroman",
    ]  # ORDERED!
    enum.sequencepats = {
        "arabic": "[0-9]+",
        "loweralpha": "[a-z]",
        "upperalpha": "[A-Z]",
        "lowerroman": "[ivxlcdm]+",
        "upperroman": "[IVXLCDM]+",
    }

    pats = {}
    pats["nonalphanum7bit"] = "[!-/:-@[-`{-~]"
    pats["alpha"] = "[a-zA-Z]"
    pats["alphanum"] = "[a-zA-Z0-9]"
    pats["alphanumplus"] = "[a-zA-Z0-9_-]"
    pats["enum"] = (
        "(%(arabic)s|%(loweralpha)s|%(upperalpha)s|%(lowerroman)s"
        "|%(upperroman)s|#)" % enum.sequencepats
    )

    for format in enum.formats:
        pats[format] = "(?P<%s>%s%s%s)" % (
            format,
            re.escape(enum.formatinfo[format].prefix),
            pats["enum"],
            re.escape(enum.formatinfo[format].suffix),
        )

    patterns = {
        "bullet": "[-+*\u2022\u2023\u2043]( +|$)",
        "enumerator": r"(%(parens)s|%(rparen)s|%(period)s)( +|$)" % pats,
    }
    for name, pat in patterns.items():
        patterns[name] = re.compile(pat)
    return patterns


PATTERNS = setup_patterns()


def get_indent(line):
    stripped_line = line.lstrip()
    indent = len(line) - len(stripped_line)
    if PATTERNS["bullet"].match(stripped_line) or PATTERNS["enumerator"].match(
        stripped_line
    ):

        indent += len(stripped_line.split(None, 1)[0]) + 1
    return indent


def get_stripped_line(line, strip_regex):
    if strip_regex:
        line = re.sub(strip_regex, "", line)
    return line


def convert(lines, strip_regex=None, yaml_strip_regex=None):
    state = STATE_TEXT
    last_text_line = ""
    last_indent = ""
    for line in lines:
        line = line.rstrip()
        if not line:
            # do not change state if the line is empty
            yield ""
        elif line.startswith("# ") or line == "#":
            if state != STATE_TEXT:
                yield ""
            line = get_stripped_line(line, strip_regex)
            line = last_text_line = line[2:]
            yield line
            last_indent = get_indent(line) * " "
            state = STATE_TEXT
        elif line == "---":
            pass
        else:
            if line.startswith("---"):
                line = line[3:]
            if state != STATE_YAML:
                if not last_text_line.endswith("::"):
                    yield last_indent + "::"
                yield ""
            line = get_stripped_line(line, yaml_strip_regex)
            yield last_indent + "  " + line
            state = STATE_YAML


def convert_text(yaml_text, strip_regex=None, yaml_strip_regex=None):
    return "\n".join(convert(yaml_text.splitlines(), strip_regex, yaml_strip_regex))


def convert_file(infilename, outfilename, strip_regex=None, yaml_strip_regex=None):
    with open(infilename) as infh:
        with open(outfilename, "w") as outfh:
            for line in convert(infh.readlines(), strip_regex, yaml_strip_regex):
                print(line.rstrip(), file=outfh)


def main(infilename, outfilename, strip_regex, yaml_strip_regex):
    if infilename == "-":
        infh = sys.stdin
    else:
        infh = open(infilename)
    if outfilename == "-":
        outfh = sys.stdout
    else:
        outfh = open(outfilename, "w")
    title = os.path.basename(infilename).removesuffix(".yaml")
    print(title, file=outfh)
    print("=" * len(title), file=outfh)
    for line in convert(infh.readlines(), strip_regex, yaml_strip_regex):
        print(line.rstrip(), file=outfh)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infilename", metavar="infile", help="YAML-file to read (`-` for stdin)"
    )
    parser.add_argument(
        "outfilename", metavar="outfile", help="rst-file to write (`-` for stdout)"
    )
    parser.add_argument(
        "--strip-regex",
        metavar="regex",
        help=(
            "Regex which will remove everything it matches. "
            "Can be used e.g. to remove fold markers from "
            "headings. "
            "Example to strip out [[[,]]] fold markers use: "
            r"'\s*(:?\[{3}|\]{3})\d?$'. "
            "Check the README for more details."
        ),
    )
    parser.add_argument(
        "--yaml-strip-regex",
        metavar="regex",
        help=(
            "Same usage as --strip-regex except that this "
            "regex substitution is preformed on the YAMl "
            "part of the file as opposed RST part."
        ),
    )
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    run()
