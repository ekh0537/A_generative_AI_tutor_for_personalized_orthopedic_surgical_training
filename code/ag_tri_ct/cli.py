import argparse
import json
from pathlib import Path

from .configuration import ExperimentConfig
from .simulation import simulate_cohort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ag-tri-ct")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--trainees", type=int, default=200)
    simulate.add_argument("--events", type=int, default=10)
    simulate.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "validate-config":
        config = ExperimentConfig.from_yaml(arguments.config)
        print(
            json.dumps(
                {
                    "effective_batch_size": config.effective_batch_size,
                    "ontology_size": config.ontology_size,
                }
            )
        )
    if arguments.command == "simulate":
        cohort = simulate_cohort(arguments.trainees, arguments.events, seed=arguments.seed)
        print(
            json.dumps(
                {
                    "trainees": len(cohort.skills),
                    "events": cohort.skills.shape[1],
                    "accuracy": float(cohort.responses.mean()),
                }
            )
        )


if __name__ == "__main__":
    main()
