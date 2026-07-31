# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.15",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # TurboVLA efficiency reproduction

    Robotic policies must turn camera images and language instructions into
    actions quickly enough to control a robot in real time. TurboVLA keeps
    this vision-and-language model compact and predicts a short action
    sequence at once. This notebook reconstructs the run-backed evidence
    behind the public reproduction report and exposes the complete measured
    sweeps for inspection.

    ## Verdict

    **Partially reproduced.** The shape-faithful architecture exceeded the
    paper's 32 Hz target and matched its memory scale on NVIDIA RTX PRO 6000
    Blackwell GPUs. Robot-task success and exact RTX 4090 performance were
    not reproduced because a trained policy checkpoint and the gated DINOv3
    weights were unavailable to the runtime.
    """)
    return


@app.cell
def _():
    campaign = {
        "paper_id": "2607.27205",
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "maximum_concurrent_gpus": 16,
        "observed_wall_hours": 11.479097,
        "terminal_runs": 33,
        "successful_runs": 26,
        "failed_runs": 5,
        "cancelled_runs": 2,
        "paper_rate_hz": 1000.0 / 31.2,
        "paper_process_vram_gb": 0.9,
    }

    endurance_rows = (
        {
            "config": "N6 / 256",
            "steps_per_replica": 1_000_000,
            "short_ms": 12.342,
            "sustained_ms": 12.349,
            "throughput_hz": 80.98,
            "process_vram_gb": 0.890625,
            "gpu_min_ms": 12.230,
            "gpu_max_ms": 12.792,
            "wall_min_ms": 14.942,
            "wall_max_ms": 77.259,
        },
        {
            "config": "N2 / 448",
            "steps_per_replica": 150_000,
            "short_ms": 10.635,
            "sustained_ms": 10.305,
            "throughput_hz": 97.04,
            "process_vram_gb": 0.886719,
            "gpu_min_ms": 10.203,
            "gpu_max_ms": 10.646,
            "wall_min_ms": 17.080,
            "wall_max_ms": 43.748,
        },
        {
            "config": "N6 / 448",
            "steps_per_replica": 100_000,
            "short_ms": 13.219,
            "sustained_ms": 12.646,
            "throughput_hz": 79.08,
            "process_vram_gb": 0.904297,
            "gpu_min_ms": 12.377,
            "gpu_max_ms": 13.393,
            "wall_min_ms": 15.425,
            "wall_max_ms": 76.680,
        },
        {
            "config": "N2 / 512",
            "steps_per_replica": 600_000,
            "short_ms": 10.412,
            "sustained_ms": 10.165,
            "throughput_hz": 98.38,
            "process_vram_gb": 0.908203,
            "gpu_min_ms": 9.956,
            "gpu_max_ms": 11.471,
            "wall_min_ms": 13.826,
            "wall_max_ms": 90.912,
        },
        {
            "config": "N6 / 512",
            "steps_per_replica": 300_000,
            "short_ms": 13.031,
            "sustained_ms": 12.575,
            "throughput_hz": 79.52,
            "process_vram_gb": 0.925781,
            "gpu_min_ms": 12.312,
            "gpu_max_ms": 14.400,
            "wall_min_ms": 20.847,
            "wall_max_ms": 88.663,
        },
    )

    depth_rows = (
        {"layers": 0, "latency_ms": 9.111, "throughput_hz": 109.76, "parameters_b": 0.201861},
        {"layers": 2, "latency_ms": 10.203, "throughput_hz": 98.01, "parameters_b": 0.206598},
        {"layers": 4, "latency_ms": 11.345, "throughput_hz": 88.14, "parameters_b": 0.211336},
        {"layers": 6, "latency_ms": 12.342, "throughput_hz": 81.03, "parameters_b": 0.216073},
        {"layers": 8, "latency_ms": 13.947, "throughput_hz": 71.72, "parameters_b": 0.220811},
    )

    memory_rows = (
        {"config": "N2 / 224", "resolution": 224, "layers": 2, "process_vram_gb": 0.878906},
        {"config": "N2 / 256", "resolution": 256, "layers": 2, "process_vram_gb": 0.878906},
        {"config": "N2 / 320", "resolution": 320, "layers": 2, "process_vram_gb": 0.878906},
        {"config": "N2 / 384", "resolution": 384, "layers": 2, "process_vram_gb": 0.882812},
        {"config": "N2 / 448", "resolution": 448, "layers": 2, "process_vram_gb": 0.886719},
        {"config": "N2 / 512", "resolution": 512, "layers": 2, "process_vram_gb": 0.908203},
        {"config": "N6 / 256", "resolution": 256, "layers": 6, "process_vram_gb": 0.890625},
        {"config": "N6 / 448", "resolution": 448, "layers": 6, "process_vram_gb": 0.904297},
        {"config": "N6 / 512", "resolution": 512, "layers": 6, "process_vram_gb": 0.925781},
    )

    text_rows = (
        {"tokens": 8, "latency_ms": 13.008, "throughput_hz": 76.88},
        {"tokens": 32, "latency_ms": 12.342, "throughput_hz": 81.03},
        {"tokens": 64, "latency_ms": 13.056, "throughput_hz": 76.59},
        {"tokens": 128, "latency_ms": 12.705, "throughput_hz": 78.71},
        {"tokens": 256, "latency_ms": 15.238, "throughput_hz": 65.63},
    )

    horizon_rows = (
        {"horizon": 8, "latency_ms": 12.616, "throughput_hz": 79.26},
        {"horizon": 12, "latency_ms": 12.342, "throughput_hz": 81.03},
        {"horizon": 15, "latency_ms": 12.524, "throughput_hz": 79.84},
        {"horizon": 50, "latency_ms": 12.771, "throughput_hz": 78.32},
    )
    return (
        campaign,
        depth_rows,
        endurance_rows,
        horizon_rows,
        memory_rows,
        text_rows,
    )


@app.cell
def _(mo):
    def bar_svg(rows, value_key, unit, target=None, color="#2563eb"):
        width, left, right = 760, 170, 70
        row_height = 48
        height = 55 + row_height * len(rows)
        values = [float(row[value_key]) for row in rows]
        maximum = max(values + ([float(target)] if target is not None else [])) * 1.12
        plot_width = width - left - right

        def x(value):
            return left + plot_width * float(value) / maximum

        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        if target is not None:
            target_x = x(target)
            pieces.append(
                f'<line x1="{target_x:.1f}" y1="12" x2="{target_x:.1f}" y2="{height - 25}" '
                'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 5"/>'
            )
            pieces.append(
                f'<text x="{target_x + 5:.1f}" y="20" font-family="sans-serif" font-size="11" '
                f'fill="#991b1b">target {target:g} {unit}</text>'
            )
        for index, row in enumerate(rows):
            y = 35 + index * row_height
            bar_width = x(row[value_key]) - left
            pieces.extend(
                [
                    f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" font-family="sans-serif" '
                    f'font-size="12" fill="#222">{row["config"]}</text>',
                    f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="25" rx="4" fill="{color}"/>',
                    f'<text x="{x(row[value_key]) + 7:.1f}" y="{y + 18}" font-family="sans-serif" '
                    f'font-size="12" font-weight="bold" fill="#1f2937">{row[value_key]:.3f} {unit}</text>',
                ]
            )
        pieces.append("</svg>")
        return mo.Html("".join(pieces))

    def line_svg(rows, x_key, y_key, x_label, y_label, color="#2563eb"):
        width, height = 760, 330
        left, right, top, bottom = 75, 30, 25, 55
        x_values = [float(row[x_key]) for row in rows]
        y_values = [float(row[y_key]) for row in rows]
        x_min, x_max = min(x_values), max(x_values)
        y_pad = max((max(y_values) - min(y_values)) * 0.2, 0.1)
        y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad

        def x(value):
            return left + (width - left - right) * (float(value) - x_min) / (x_max - x_min)

        def y(value):
            return top + (height - top - bottom) * (y_max - float(value)) / (y_max - y_min)

        points = " ".join(f"{x(row[x_key]):.1f},{y(row[y_key]):.1f}" for row in rows)
        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>',
        ]
        for row in rows:
            pieces.extend(
                [
                    f'<circle cx="{x(row[x_key]):.1f}" cy="{y(row[y_key]):.1f}" r="6" fill="{color}" '
                    'stroke="white" stroke-width="2"/>',
                    f'<text x="{x(row[x_key]):.1f}" y="{y(row[y_key]) - 12:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" fill="#1f2937">{row[y_key]:.3f}</text>',
                    f'<text x="{x(row[x_key]):.1f}" y="{height - bottom + 20}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" fill="#444">{row[x_key]}</text>',
                ]
            )
        pieces.extend(
            [
                f'<text x="{(left + width - right) / 2}" y="{height - 10}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{x_label}</text>',
                f'<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" '
                f'font-family="sans-serif" font-size="12">{y_label}</text>',
                "</svg>",
            ]
        )
        return mo.Html("".join(pieces))

    def memory_svg(rows):
        width, height = 760, 350
        left, right, top, bottom = 75, 30, 25, 55

        def x(value):
            return left + (width - left - right) * (float(value) - 224) / (512 - 224)

        def y(value):
            return top + (height - top - bottom) * (0.94 - float(value)) / (0.94 - 0.86)

        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{left}" y1="{y(0.9):.1f}" x2="{width - right}" y2="{y(0.9):.1f}" '
            'stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<text x="{width - right}" y="{y(0.9) - 7:.1f}" text-anchor="end" font-family="sans-serif" '
            'font-size="11" fill="#991b1b">0.9 GB paper scale</text>',
        ]
        for layers, color in ((2, "#059669"), (6, "#7c3aed")):
            series = sorted((row for row in rows if row["layers"] == layers), key=lambda row: row["resolution"])
            points = " ".join(f'{x(row["resolution"]):.1f},{y(row["process_vram_gb"]):.1f}' for row in series)
            pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
            for row in series:
                pieces.extend(
                    [
                        f'<circle cx="{x(row["resolution"]):.1f}" cy="{y(row["process_vram_gb"]):.1f}" '
                        f'r="6" fill="{color}" stroke="white" stroke-width="2"/>',
                        f'<text x="{x(row["resolution"]):.1f}" y="{y(row["process_vram_gb"]) - 11:.1f}" '
                        f'text-anchor="middle" font-family="sans-serif" font-size="10" fill="{color}">'
                        f'{row["process_vram_gb"]:.3f}</text>',
                    ]
                )
        pieces.extend(
            [
                f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
                f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" '
                'font-size="12">Pixels per camera view</text>',
                f'<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" '
                'font-family="sans-serif" font-size="12">Process memory (GB)</text>',
                '<text x="95" y="35" font-family="sans-serif" font-size="11" fill="#059669">N2</text>',
                '<text x="130" y="35" font-family="sans-serif" font-size="11" fill="#7c3aed">N6</text>',
                "</svg>",
            ]
        )
        return mo.Html("".join(pieces))

    def pair_svg(rows):
        width, height = 760, 345
        left_x, right_x = 250, 540
        top, bottom = 25, 65
        values = [row[key] for row in rows for key in ("short_ms", "sustained_ms")]
        y_min, y_max = min(values) - 0.5, max(values) + 0.5
        colors = ("#2563eb", "#059669", "#7c3aed", "#dc2626", "#0891b2")

        def y(value):
            return top + (height - top - bottom) * (y_max - float(value)) / (y_max - y_min)

        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        for row, color in zip(rows, colors):
            pieces.extend(
                [
                    f'<line x1="{left_x}" y1="{y(row["short_ms"]):.1f}" x2="{right_x}" '
                    f'y2="{y(row["sustained_ms"]):.1f}" stroke="{color}" stroke-width="3"/>',
                    f'<circle cx="{left_x}" cy="{y(row["short_ms"]):.1f}" r="6" fill="{color}"/>',
                    f'<circle cx="{right_x}" cy="{y(row["sustained_ms"]):.1f}" r="6" fill="{color}"/>',
                    f'<text x="{left_x - 10}" y="{y(row["short_ms"]) + 4:.1f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="10" fill="{color}">{row["config"]} {row["short_ms"]:.3f}</text>',
                    f'<text x="{right_x + 10}" y="{y(row["sustained_ms"]) + 4:.1f}" '
                    f'font-family="sans-serif" font-size="10" fill="{color}">{row["sustained_ms"]:.3f}</text>',
                ]
            )
        pieces.extend(
            [
                f'<text x="{left_x}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="12">short</text>',
                f'<text x="{right_x}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="12">sustained</text>',
                "</svg>",
            ]
        )
        return mo.Html("".join(pieces))

    def range_svg(rows):
        width, height = 760, 285
        left, right = 150, 40

        def x(value):
            return left + (width - left - right) * float(value) / 100.0

        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        for index, row in enumerate(rows):
            y_gpu = 40 + index * 78
            y_wall = y_gpu + 27
            pieces.extend(
                [
                    f'<text x="{left - 12}" y="{y_gpu + 17}" text-anchor="end" font-family="sans-serif" '
                    f'font-size="12">{row["config"]}</text>',
                    f'<line x1="{x(row["gpu_min_ms"]):.1f}" y1="{y_gpu}" x2="{x(row["gpu_max_ms"]):.1f}" '
                    'y2="{y_gpu}" stroke="#2563eb" stroke-width="9" stroke-linecap="round"/>'.format(y_gpu=y_gpu),
                    f'<line x1="{x(row["wall_min_ms"]):.1f}" y1="{y_wall}" x2="{x(row["wall_max_ms"]):.1f}" '
                    'y2="{y_wall}" stroke="#f59e0b" stroke-width="9" stroke-linecap="round"/>'.format(y_wall=y_wall),
                    f'<text x="{x(row["gpu_max_ms"]) + 8:.1f}" y="{y_gpu + 4}" font-family="sans-serif" '
                    f'font-size="10">GPU {row["gpu_min_ms"]:.1f}–{row["gpu_max_ms"]:.1f}</text>',
                    f'<text x="{x(row["wall_max_ms"]) - 4:.1f}" y="{y_wall + 4}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="10">wall {row["wall_min_ms"]:.1f}–{row["wall_max_ms"]:.1f}</text>',
                ]
            )
        pieces.append("</svg>")
        return mo.Html("".join(pieces))

    return bar_svg, line_svg, memory_svg, pair_svg, range_svg


@app.cell
def _(campaign, endurance_rows, mo):
    canonical = endurance_rows[0]
    verdict_card = mo.callout(
        mo.md(
            f"""
            **Run-backed headline.** {campaign["successful_runs"]} successful
            Kubernetes runs have terminal measurement summaries. The canonical
            N6/256 endurance run sustained **{canonical["throughput_hz"]:.2f} Hz**
            at **{canonical["process_vram_gb"]:.3f} GB** process memory—about
            **{canonical["throughput_hz"] / 32:.1f}×** the 32 Hz target.
            """
        ),
        kind="success",
    )
    verdict_card
    return


@app.cell
def _(mo):
    selected_metric = mo.ui.dropdown(
        options=["throughput_hz", "sustained_ms", "process_vram_gb"],
        value="throughput_hz",
        label="Endurance metric",
    )
    selected_metric
    return (selected_metric,)


@app.cell
def _(bar_svg, campaign, endurance_rows, selected_metric):
    metric_specs = {
        "throughput_hz": ("Hz", 32.0, "#2563eb"),
        "sustained_ms": ("ms", 31.2, "#7c3aed"),
        "process_vram_gb": ("GB", campaign["paper_process_vram_gb"], "#059669"),
    }
    selected_unit, selected_target, selected_color = metric_specs[selected_metric.value]
    endurance_chart = bar_svg(
        endurance_rows,
        selected_metric.value,
        selected_unit,
        target=selected_target,
        color=selected_color,
    )
    return (endurance_chart,)


@app.cell
def _(endurance_chart, endurance_rows, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Sustained efficiency

                Use the selector above to compare throughput, GPU latency, or
                process memory. The 0.9 GB memory line is a reported scale, not
                a universal hard limit; values just above it remain below the
                paper title's broader 1 GB threshold.
                """
            ),
            endurance_chart,
            mo.ui.table(list(endurance_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(depth_rows, line_svg, mo):
    depth_chart = line_svg(
        depth_rows,
        "layers",
        "latency_ms",
        "Interaction layers",
        "Median model latency (ms)",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Interaction depth is the main latency lever

                The matched 256-pixel sweep isolates the number of
                vision-language interaction layers. Moving from zero to eight
                layers adds 4.836 ms and 18.95 million parameters, while every
                measured depth still clears 71 Hz.
                """
            ),
            depth_chart,
            mo.ui.table(list(depth_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(memory_rows, memory_svg, mo):
    memory_chart = memory_svg(memory_rows)
    mo.vstack(
        [
            mo.md(
                """
                ## Resolution and the memory boundary

                The two-layer variant stays below 0.9 GB through 448 pixels per
                view. Canonical N6 is below that line at 256 pixels, then reaches
                0.904 GB at 448 and 0.926 GB at 512. This is why the report
                distinguishes the reported 0.9 GB scale from a strict
                under-0.9-GB claim.
                """
            ),
            memory_chart,
            mo.ui.table(list(memory_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(endurance_rows, mo, pair_svg):
    stability_chart = pair_svg(endurance_rows)
    maximum_change_ms = max(abs(row["sustained_ms"] - row["short_ms"]) for row in endurance_rows)
    mo.vstack(
        [
            mo.md(
                f"""
                ## Endurance does not degrade GPU latency

                Across all five matched configurations, the largest absolute
                difference between the short and sustained median is
                **{maximum_change_ms:.3f} ms**. The N6/256 arm is especially
                strong evidence: eight replicas each completed one million
                model forwards.
                """
            ),
            stability_chart,
        ]
    )
    return


@app.cell
def _(endurance_rows, mo, range_svg):
    diagnostic_rows = tuple(
        row for row in endurance_rows if row["config"] in {"N6 / 256", "N2 / 512", "N6 / 512"}
    )
    timing_chart = range_svg(diagnostic_rows)
    mo.vstack(
        [
            mo.md(
                """
                ## GPU time and wall time answer different questions

                Blue segments are the range of per-replica GPU-event medians.
                Orange segments include preprocessing and CPU scheduling.
                Narrow blue ranges establish stable model execution; broad
                orange ranges diagnose cluster-side contention rather than a
                slower neural network.
                """
            ),
            timing_chart,
        ]
    )
    return


@app.cell
def _(horizon_rows, mo, text_rows):
    minimum_secondary_rate = min(
        [row["throughput_hz"] for row in text_rows]
        + [row["throughput_hz"] for row in horizon_rows]
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## Secondary sweeps

                Action horizons from 8 to 50 remain tightly grouped around
                78–81 Hz. Cached text lengths through 128 tokens also have
                modest cost; 256 tokens is the slowest secondary condition at
                **{minimum_secondary_rate:.2f} Hz**, still more than twice the
                real-time target.
                """
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md("**Cached text**"), mo.ui.table(list(text_rows), pagination=False)]),
                    mo.vstack([mo.md("**Action horizon**"), mo.ui.table(list(horizon_rows), pagination=False)]),
                ],
                widths="equal",
            ),
        ]
    )
    return


@app.cell
def _(campaign, mo):
    mo.md(
        f"""
        ## Evidence audit and limitations

        The campaign consumed **{campaign["observed_wall_hours"]:.6f} observed
        Kubernetes wall hours** with at most
        **{campaign["maximum_concurrent_gpus"]} GPUs** allocated concurrently.
        Its {campaign["terminal_runs"]} terminal runs comprise
        {campaign["successful_runs"]} successful measurement runs,
        {campaign["failed_runs"]} failed setup or aggregation attempts, and
        {campaign["cancelled_runs"]} cancelled superseded attempts. Failed and
        cancelled runs contribute no measurements.

        The policy and DINOv3 values were random but shape-faithful. This
        preserves dense parameter counts, tensor geometry, allocation, and
        executed kernels, but it cannot test action quality or closed-loop
        robot success. The hardware is RTX PRO 6000 Blackwell rather than RTX
        4090, so the measured rates are evidence for the architecture's
        efficiency—not direct 4090 benchmark numbers.
        """
    )
    return


if __name__ == "__main__":
    app.run()
