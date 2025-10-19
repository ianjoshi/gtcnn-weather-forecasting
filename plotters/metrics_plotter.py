import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict


def parse_report(report_path: Path) -> Dict[str, float]:
    """
    Parse a report file and extract metrics.
    
    Args:
        report_path (Path): Path to the report text file.
    
    Returns:
        Dict[str, float]: Dictionary of metric names to values.
    """
    metrics = {}
    with open(report_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if ":" in line and not line.startswith("-"):
                parts = line.split(":")
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    metric_value = parts[1].strip()
                    try:
                        metrics[metric_name] = float(metric_value)
                    except ValueError:
                        pass
    return metrics


def plot_metrics_comparison(report_dir: Path, model_mapping: Dict[str, str], output_dir: Path = None):
    """
    Generate comparison plots for each metric across all models.
    
    Args:
        report_dir (Path): Directory containing report_*.txt files.
        model_mapping (Dict[str, str]): Mapping from report filename (without extension) to model display name.
                                       Example: {"report_gtcnn": "GTCNN", "report_persistence": "Persistence"}
        output_dir (Path, optional): Directory to save plots. Defaults to report_dir.
    """
    report_dir = Path(report_dir)
    if output_dir is None:
        output_dir = report_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all reports
    report_files = sorted(report_dir.glob("report_*.txt"))
    
    if not report_files:
        print(f"No report files found in {report_dir}")
        return
    
    # Parse all reports
    all_metrics = {}
    for report_path in report_files:
        report_name = report_path.stem  # e.g., "report_gtcnn"
        
        # Look up display name in mapping
        if report_name in model_mapping:
            model_name = model_mapping[report_name]
        else:
            # Fallback: use the model type from filename
            model_name = report_name.replace("report_", "").upper()
        
        metrics = parse_report(report_path)
        if metrics:
            all_metrics[model_name] = metrics
    
    if not all_metrics:
        print("No metrics found in report files.")
        return
    
    # Collect all unique metric names
    all_metric_names = set()
    for metrics in all_metrics.values():
        all_metric_names.update(metrics.keys())
    
    # Create one plot per metric
    for metric_name in sorted(all_metric_names):
        models = []
        values = []
        
        for model_name in sorted(all_metrics.keys()):
            if metric_name in all_metrics[model_name]:
                models.append(model_name)
                values.append(all_metrics[model_name][metric_name])
        
        if not values:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison Across Models', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"comparison_{metric_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_name} comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate metric comparison plots across models")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root directory (reports will be in root_dir/reports, plots in root_dir/plots)",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="Model display names in format: report_name1:DisplayName1 report_name2:DisplayName2 ...",
    )
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    report_dir = root_dir / "reports"
    output_dir = root_dir / "plots"
    
    # Build model mapping
    model_mapping = {}
    if args.model_names:
        for mapping in args.model_names:
            if ":" in mapping:
                report_name, display_name = mapping.split(":", 1)
                model_mapping[report_name] = display_name
    
    plot_metrics_comparison(
        report_dir=report_dir,
        model_mapping=model_mapping,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()