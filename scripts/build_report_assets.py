"""Build report-ready figures and tables from saved artifacts."""

from oulad_causal.dag import write_dag_artifacts


def main() -> None:
    """Build deterministic report assets that do not require model fitting."""

    paths = write_dag_artifacts()
    print("Wrote DAG artifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
