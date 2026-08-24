from .benchmark import Benchmark


def _stage_rows(stage_times):
    """Build HTML table rows for pipeline stages."""
    rows = []
    for stage, duration in stage_times.items():
        label = stage.replace("_", " ").title()
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{duration:.2f}</td></tr>"
        )
    return "\n".join(rows)


def _stage_pct_rows(stage_times, stage_percentages):
    """Build HTML table rows with percentage column."""
    rows = []
    for stage, duration in stage_times.items():
        label = stage.replace("_", " ").title()
        pct = stage_percentages.get(stage, 0)
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{duration:.2f}</td>"
            f"<td>{pct:.2f}%</td></tr>"
        )
    return "\n".join(rows)


def _analysis_time_rows(analysis_times):
    """Build HTML table rows for per-image analysis timing."""
    rows = []
    for image_name, duration in analysis_times.items():
        rows.append(
            f"<tr><td>{image_name}</td>"
            f"<td>{duration:.2f}</td></tr>"
        )
    return "\n".join(rows)


def _error_section(errors):
    """Build HTML for the errors section."""
    if not errors:
        return "<p>No errors reported.</p>"

    items = "".join(f"<li>{error}</li>" for error in errors)
    return f"<ul>{items}</ul>"


def generate_benchmark_report(benchmark: Benchmark) -> str:
    """
    Generate an HTML benchmark report for one document.
    """

    analysis_times = list(benchmark.analysis_times.values())

    if analysis_times:
        analysis_avg = sum(analysis_times) / len(analysis_times)
        analysis_min = min(analysis_times)
        analysis_max = max(analysis_times)
    else:
        analysis_avg = 0
        analysis_min = 0
        analysis_max = 0

    stage_percentages = {}

    if benchmark.total_time > 0:
        for stage, duration in benchmark.stage_times.items():
            stage_percentages[stage] = (
                duration / benchmark.total_time
            ) * 100

    stages_html = _stage_rows(benchmark.stage_times)
    analysis_html = _analysis_time_rows(benchmark.analysis_times)
    distribution_html = _stage_pct_rows(
        benchmark.stage_times, stage_percentages
    )
    errors_html = _error_section(benchmark.errors)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Document Processing Benchmark</title>
    </head>

    <body>

        <h1>Document Processing Benchmark</h1>

        <h2>Document Information</h2>

        <p><strong>Document:</strong> {benchmark.document_name}</p>
        <p><strong>Type:</strong> {benchmark.document_type}</p>
        <p><strong>File Size:</strong> {benchmark.file_size_mb:.2f} MB</p>

        <h2>Performance</h2>

        <p>
            <strong>Total Processing Time:</strong>
            {benchmark.total_time:.2f} seconds
        </p>

        <h2>Pipeline Stage Timing</h2>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr>
                <th>Stage</th>
                <th>Time (seconds)</th>
            </tr>
            {stages_html}
        </table>

        <h2>Image Statistics</h2>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr>
                <th>Metric</th>
                <th>Count</th>
            </tr>

            <tr>
                <td>Images Detected</td>
                <td>{benchmark.images_detected}</td>
            </tr>

            <tr>
                <td>Images Processed</td>
                <td>{benchmark.images_processed}</td>
            </tr>

            <tr>
                <td>Images Skipped</td>
                <td>{benchmark.images_skipped}</td>
            </tr>
        </table>

        <h2>Image Analysis Statistics</h2>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr>
                <th>Metric</th>
                <th>Count</th>
            </tr>

            <tr>
                <td>Images Analyzed</td>
                <td>{benchmark.images_analyzed}</td>
            </tr>

            <tr>
                <td>Images from Cache</td>
                <td>{benchmark.images_cached}</td>
            </tr>

            <tr>
                <td>Images Skipped</td>
                <td>{benchmark.images_skipped}</td>
            </tr>

            <tr>
                <td>Analysis Failures</td>
                <td>{benchmark.images_failed}</td>
            </tr>
        </table>

        <h2>Per-Image Analysis Timing</h2>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr>
                <th>Image</th>
                <th>Analysis Time (seconds)</th>
            </tr>
            {analysis_html}
        </table>

        <h2>Analysis Timing Summary</h2>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr>
                <th>Metric</th>
                <th>Time (seconds)</th>
            </tr>

            <tr>
                <td>Average Analysis Time</td>
                <td>{analysis_avg:.2f}</td>
            </tr>

            <tr>
                <td>Fastest Analysis</td>
                <td>{analysis_min:.2f}</td>
            </tr>

            <tr>
                <td>Slowest Analysis</td>
                <td>{analysis_max:.2f}</td>
            </tr>
        </table>

        <h2>Pipeline Time Distribution</h2>

        <table border="1" cellpadding="6" cellspacing="0">
            <tr>
                <th>Stage</th>
                <th>Time (seconds)</th>
                <th>Percentage of Total</th>
            </tr>
            {distribution_html}
        </table>

        <h2>Errors and Failures</h2>

        {errors_html}

    </body>
    </html>
    """
