#!/usr/bin/env python3
import argparse
import json

from spatial_forcing.cache import inspect_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    args = parser.parse_args()
    print(json.dumps(inspect_cache(args.cache), indent=2, ensure_ascii=False))

