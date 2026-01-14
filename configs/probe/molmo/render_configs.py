from __future__ import annotations

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

DATASETS = [
    "robospatial_compatibility",
    "robospatial_configuration",
]

DB_VARIANTS = [
    "Mixed",
    "Mixed80k",
    "Mixed400k",
    "Mixed800k",
    "prism",
    "RefSpatial",
    "RoboSpatial",
    "SAT",
    "SPAR-7M",
    "Spatial457",
]

LAYERS = range(32)  # 0..31

TEMPLATE_FILE = "config.yaml.j2"
OUT_ROOT = Path("configs/probe/molmo")


def main() -> None:
    script_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(script_dir), autoescape=False)
    tmpl = env.get_template(TEMPLATE_FILE)

    count = 0
    for dataset in DATASETS:
        for db_variant in DB_VARIANTS:
            for layer_idx in LAYERS:
                rendered = tmpl.render(
                    dataset=dataset,
                    db_variant=db_variant,
                    layer_idx=layer_idx,
                )

                out_dir = script_dir / dataset / db_variant
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"l{layer_idx}.yaml"
                out_path.write_text(rendered)

                count += 1

    print(f"Wrote {count} configs under {script_dir}/")


if __name__ == "__main__":
    main()
