"""
Patch instructions for evaluate.py — apply manually or use as reference.

TWO CHANGES ONLY:
  1. _scan_pareto(): check for args.single_episode and override path_meta_list
  2. main():         add --single-episode argument

No other changes. All evaluation logic (HV, Pareto, objectives) is unchanged.
"""

# ============================================================
# CHANGE 1: in _scan_pareto(args), after the block that builds
# path_meta_list (both recursive and flat branches), ADD:
# ============================================================

PATCH_SCAN_PARETO = '''
    # --single-episode: override to exactly 1 file for quick visual check
    if getattr(args, "single_episode", None):
        if not os.path.exists(args.single_episode):
            raise FileNotFoundError(
                f"--single-episode not found: {args.single_episode}"
            )
        path_meta_list = [
            (
                args.single_episode,
                {
                    "topology": "single",
                    "allocation": "single",
                    "difficulty": "single",
                    "test_index": 0,
                },
            )
        ]
        print(f"[evaluate] single-episode mode: {args.single_episode}")
'''

# Insert location: after the `if not path_meta_list:` check in _scan_pareto,
# i.e. right before the line:
#   n_pts = args.pareto_points

# ============================================================
# CHANGE 2: in main(), add one argument to the parser:
# ============================================================

PATCH_MAIN_ARG = '''
    parser.add_argument(
        "--single-episode",
        type=str,
        default="",
        help="Path to a single test episode JSON. When set with --pareto-scan, "
             "evaluates only that one file for quick visual/HV check. "
             "Full evaluation (all 10 test files) runs when this flag is omitted.",
    )
'''
# Insert location: at the end of the argument block in main(), before
# args = parser.parse_args()


# ============================================================
# FULL REPLACEMENT of the two relevant code sections follows,
# for copy-paste if preferred over manual patching.
# ============================================================

# ---------- _scan_pareto: replace the line ----------
#   if not path_meta_list:
#       raise FileNotFoundError(f"No episodes found in {args.data_dir}")
# with:

REPLACEMENT_NOT_FOUND_BLOCK = '''
    if not path_meta_list:
        raise FileNotFoundError(f"No episodes found in {args.data_dir}")

    # --single-episode: override to exactly 1 file for quick visual check
    if getattr(args, "single_episode", None):
        if not os.path.exists(args.single_episode):
            raise FileNotFoundError(
                f"--single-episode not found: {args.single_episode}"
            )
        path_meta_list = [
            (
                args.single_episode,
                {
                    "topology": "single",
                    "allocation": "single",
                    "difficulty": "single",
                    "test_index": 0,
                },
            )
        ]
        print(f"[evaluate] single-episode mode: {args.single_episode}")
'''

# ---------- main(): replace ----------
#   parser.add_argument("--hv-eps", type=float, default=1e-2)
#   parser.add_argument(
#       "--recursive", ...
#   )
# with the same block + the new argument:

REPLACEMENT_MAIN_ARGS_TAIL = '''
    parser.add_argument("--hv-eps", type=float, default=1e-2)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover episodes under --data-dir with "
        "{topology}/{allocation}/{difficulty}/ structure (for data/all_tests/).",
    )
    parser.add_argument(
        "--single-episode",
        type=str,
        default="",
        help="Path to a single test episode JSON. When set with --pareto-scan, "
             "evaluates only that one file for quick visual/HV check. "
             "Full evaluation (all 10 test files) runs when this flag is omitted.",
    )
'''
