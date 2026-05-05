import yaml

from simsit import simulate


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run a SImSiT image simulation from a YAML config.")
    parser.add_argument("config", type=str, help="Path to simulation configuration YAML file.")
    parser.add_argument(
        "--make-sky-flat",
        action="store_true",
        help="Also generate a sky flat from the simulated images.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    simulate.simulate(config)

    if args.make_sky_flat:
        simulate.make_sky_flat(args.config)


if __name__ == "__main__":
    main()