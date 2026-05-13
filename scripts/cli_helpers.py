from argparse import ArgumentParser


def add_task_argument(parser, task_choices, default="all"):
    parser.add_argument(
        "--task",
        choices=[default, *sorted(task_choices)],
        default=default,
        help="Task preset to run. Defaults to all tasks.",
    )


def add_passthrough_task_argument(parser, default="all"):
    parser.add_argument(
        "--task",
        default=default,
        help="Task preset to pass through to each training script. Defaults to all.",
    )


def add_metadata_feature_arguments(parser):
    parser.add_argument(
        "--include-location", action="store_true", help="Include Location metadata."
    )
    parser.add_argument(
        "--include-gender", action="store_true", help="Include Gender metadata."
    )


def add_metadata_variant_argument(parser, metadata_variants):
    parser.add_argument(
        "--metadata",
        nargs="+",
        choices=list(metadata_variants),
        default=list(metadata_variants),
        help="Metadata variants to run. Defaults to all variants.",
    )


def add_compare_arguments(parser):
    parser.add_argument(
        "--task",
        help="Optional task name filter, such as lungs_only__lung.",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "f1_macro_mean",
            "accuracy_mean",
            "precision_macro_mean",
            "recall_macro_mean",
        ],
        default="f1_macro_mean",
        help="Metric to sort by. Defaults to f1_macro_mean.",
    )


def make_parser(description):
    return ArgumentParser(description=description)
