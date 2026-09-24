"""Rebuild the six manuscript figures from a matching released data package.

All inputs are read-only. Outputs must be a new directory outside the data and
code components. --check-only validates source relationships without rendering.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from sparrow_figures.resources import ASSETS, COMPONENT
from sparrow_figures.validation import (
    check_assets,
    check_spectra,
    check_tables,
    require,
    rows,
    sha256,
)


def export_figures(package, output, counts, selected, spectra):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sparrow_figures import (
        equipment,
        flow,
        hierarchy,
        site,
        stft,
        structures,
        style,
        variant,
    )

    reports = []

    def save(figure, number, name):
        figure.canvas.draw()
        target = output / f"figure{number}"
        target.mkdir()
        for extension in ["pdf", "svg", "png"]:
            figure.savefig(target / f"{name}.{extension}", dpi=400)
        reports.append(
            {
                "figure": number,
                "name": name,
                "size_inches": figure.get_size_inches().tolist(),
                "axes": len(figure.axes),
                "outputs": {
                    ext: sha256(target / f"{name}.{ext}")
                    for ext in ["pdf", "svg", "png"]
                },
            }
        )
        plt.close(figure)

    with plt.rc_context():
        save(site.main(), 1, "site")
    with plt.rc_context():
        save(equipment.main(), 2, "equipment")
    with plt.rc_context():
        style.setup()
        figure = flow.build(counts)
        require(len(flow.GEOMETRY["nodes"]) == 12, "Flow geometry")
        save(figure, 3, "flow")
    with plt.rc_context():
        save(hierarchy.main(package / "metadata"), 4, "hierarchy")
    with plt.rc_context():
        save(
            structures.draw([r for r in selected if r["figure"] == "figure5"], spectra),
            5,
            "structures",
        )
    with plt.rc_context():
        variant.style()
        rows_ = [row for row in selected if row["figure"] == "figure6"]
        panels = {
            row["candidate_id"]: (spectra[row["stable_id"]], row) for row in rows_
        }
        titles = ["Main form (single_group_08)", "Variant (single_group_08_v1)"]
        figure = variant.spectrogram_figure(
            stft,
            panels,
            [(r["candidate_id"], title) for r, title in zip(rows_, titles)],
            (4.8, 1.9),
            "variant",
        )
        for axis in figure.axes[:-1]:
            axis.set_xlim(0, 0.4)
            axis.set_xticks([0, 0.2, 0.4])
            axis.set_xticklabels(["0", "0.2", "0.4"])
            axis.images[0].set_clim(-45, 0)
        colorbar = figure.axes[-1]._colorbar
        colorbar.mappable.set_clim(-45, 0)
        colorbar.set_ticks([-45, -20, 0])
        colorbar.set_ticklabels(["−45", "−20", "0"])
        colorbar.ax.set_ylim(-45, 0)
        save(figure, 6, "variant")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    package, output = (args.data_package.resolve(), args.output.resolve())
    require(
        not output.is_relative_to(package) and (not output.is_relative_to(COMPONENT)),
        "Output must be outside read-only data/code inputs",
    )
    require(not output.exists(), "Output directory must not already exist")
    assets = check_assets()
    counts, data_check = check_tables(package)
    from sparrow_figures.hierarchy_data import validated_data

    _, actual_rows = validated_data(package / "metadata")
    require(
        [{k: str(v) for k, v in row.items()} for row in actual_rows]
        == rows(ASSETS / "hierarchy_leaves.csv"),
        "Hierarchy source table",
    )
    selected, spectra, spectral_checks = check_spectra(package)
    output.mkdir(parents=True)
    report = {
        "status": "passed",
        "mode": "check_only" if args.check_only else "render",
        "validated_assets": assets,
        "data": data_check,
        "spectra": spectral_checks,
        "new_scientific_analysis": False,
        "figures": (
            []
            if args.check_only
            else export_figures(package, output, counts, selected, spectra)
        ),
    }
    (output / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "figures": len(report["figures"]),
                "assets": assets,
                "raven_tables": 138,
                "syllables_checked": len(spectral_checks),
            }
        )
    )


if __name__ == "__main__":
    main()
