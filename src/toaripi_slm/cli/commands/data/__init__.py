"""
Data management and validation commands.
"""

import click
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import csv

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt, IntPrompt

from ...context import get_context

console = Console()

@click.group()
def data():
    """Data validation, preparation, and management commands."""
    pass

@data.command()
@click.argument("file_path", type=Path)
@click.option("--strict", is_flag=True, help="Strict validation mode")
@click.option("--fix", is_flag=True, help="Attempt to fix common issues")
@click.option("--report", type=Path, help="Save validation report to file")
def validate(file_path, strict, fix, report):
    """Validate parallel training data format and quality."""
    ctx = get_context()
    
    console.print("📊 [bold blue]Data Validation[/bold blue]\n")
    
    if not file_path.exists():
        console.print(f"❌ File not found: {file_path}")
        return
    
    console.print(f"Validating: [cyan]{file_path}[/cyan]")
    
    # Perform validation
    validation_result = validate_data_file(file_path, strict=strict, fix=fix)
    
    # Display results
    display_validation_results(validation_result)
    
    # Save report if requested
    if report:
        save_validation_report(validation_result, report)
        console.print(f"📄 Report saved to: [cyan]{report}[/cyan]")

def validate_data_file(file_path: Path, strict: bool = False, fix: bool = False) -> Dict[str, Any]:
    """Validate a parallel data file."""
    
    results = {
        "file_path": str(file_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "stats": {},
        "fixed_issues": []
    }
    
    try:
        # Load data
        console.print("📋 Loading data file...")
        df = pd.read_csv(file_path)
        
        # Basic structure validation
        required_columns = ["english", "toaripi"]
        optional_columns = ["verse_id", "book", "chapter"]
        
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            results["errors"].append(f"Missing required columns: {missing_required}")
        
        # Data quality checks
        console.print("🔍 Checking data quality...")
        
        # Check for empty rows
        empty_english = df["english"].isna().sum() if "english" in df.columns else 0
        empty_toaripi = df["toaripi"].isna().sum() if "toaripi" in df.columns else 0
        
        if empty_english > 0:
            if strict:
                results["errors"].append(f"{empty_english} rows with empty English text")
            else:
                results["warnings"].append(f"{empty_english} rows with empty English text")
        
        if empty_toaripi > 0:
            if strict:
                results["errors"].append(f"{empty_toaripi} rows with empty Toaripi text") 
            else:
                results["warnings"].append(f"{empty_toaripi} rows with empty Toaripi text")
        
        # Length checks
        if "english" in df.columns and "toaripi" in df.columns:
            # Check for very short or very long texts
            short_english = (df["english"].str.len() < 5).sum()
            long_english = (df["english"].str.len() > 500).sum()
            short_toaripi = (df["toaripi"].str.len() < 3).sum()
            long_toaripi = (df["toaripi"].str.len() > 500).sum()
            
            if short_english > 0:
                results["warnings"].append(f"{short_english} very short English texts (<5 chars)")
            if long_english > 0:
                results["warnings"].append(f"{long_english} very long English texts (>500 chars)")
            if short_toaripi > 0:
                results["warnings"].append(f"{short_toaripi} very short Toaripi texts (<3 chars)")
            if long_toaripi > 0:
                results["warnings"].append(f"{long_toaripi} very long Toaripi texts (>500 chars)")
        
        # Character encoding checks
        console.print("🔤 Checking character encoding...")
        
        # Check for unusual characters
        if "toaripi" in df.columns:
            toaripi_text = " ".join(df["toaripi"].dropna())
            unusual_chars = set(char for char in toaripi_text if ord(char) > 127)
            if unusual_chars and len(unusual_chars) > 10:
                results["warnings"].append(f"Found {len(unusual_chars)} non-ASCII characters - verify encoding")
        
        # Statistical analysis
        results["stats"] = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "avg_english_length": df["english"].str.len().mean() if "english" in df.columns else 0,
            "avg_toaripi_length": df["toaripi"].str.len().mean() if "toaripi" in df.columns else 0,
            "empty_rows": empty_english + empty_toaripi,
        }
        
        # Apply fixes if requested
        if fix and (empty_english > 0 or empty_toaripi > 0):
            original_len = len(df)
            df = df.dropna(subset=["english", "toaripi"])
            removed = original_len - len(df)
            if removed > 0:
                results["fixed_issues"].append(f"Removed {removed} rows with empty text")
                # Save fixed file
                fixed_path = file_path.with_suffix(".fixed.csv")
                df.to_csv(fixed_path, index=False)
                results["fixed_issues"].append(f"Saved cleaned data to {fixed_path}")
        
        # Final validation
        if not results["errors"]:
            results["valid"] = True
            
    except Exception as e:
        results["errors"].append(f"Failed to load file: {str(e)}")
    
    return results

def display_validation_results(results: Dict[str, Any]):
    """Display validation results in a formatted way."""
    
    # Status panel
    status = "✅ VALID" if results["valid"] else "❌ INVALID"
    status_color = "green" if results["valid"] else "red"
    
    status_panel = Panel(
        f"""
        [bold {status_color}]{status}[/bold {status_color}]
        
        File: {results["file_path"]}
        Total rows: {results["stats"].get("total_rows", 0)}
        Errors: {len(results["errors"])}
        Warnings: {len(results["warnings"])}
        """,
        title="📊 Validation Results",
        border_style=status_color
    )
    console.print(status_panel)
    
    # Statistics
    if results["stats"]:
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats = results["stats"]
        stats_table.add_row("Total Rows", str(stats.get("total_rows", 0)))
        stats_table.add_row("Columns", ", ".join(stats.get("columns", [])))
        stats_table.add_row("Avg English Length", f"{stats.get('avg_english_length', 0):.1f} chars")
        stats_table.add_row("Avg Toaripi Length", f"{stats.get('avg_toaripi_length', 0):.1f} chars")
        
        console.print("\n📈 Data Statistics:")
        console.print(stats_table)
    
    # Errors
    if results["errors"]:
        console.print("\n❌ [bold red]Errors:[/bold red]")
        for error in results["errors"]:
            console.print(f"   • {error}")
    
    # Warnings
    if results["warnings"]:
        console.print("\n⚠️  [bold yellow]Warnings:[/bold yellow]")
        for warning in results["warnings"]:
            console.print(f"   • {warning}")
    
    # Fixed issues
    if results["fixed_issues"]:
        console.print("\n🔧 [bold blue]Fixed Issues:[/bold blue]")
        for fix in results["fixed_issues"]:
            console.print(f"   • {fix}")

@data.command()
@click.argument("input_file", type=Path)
@click.option("--output-dir", type=Path, default="data/processed", help="Output directory")
@click.option("--train-ratio", type=float, default=0.8, help="Training data ratio")
@click.option("--val-ratio", type=float, default=0.1, help="Validation data ratio")
@click.option("--test-ratio", type=float, default=0.1, help="Test data ratio")
@click.option("--shuffle", is_flag=True, default=True, help="Shuffle data before splitting")
def split(input_file, output_dir, train_ratio, val_ratio, test_ratio, shuffle):
    """Split parallel data into train/validation/test sets."""
    ctx = get_context()
    
    console.print("✂️  [bold blue]Data Splitting[/bold blue]\n")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        console.print(f"❌ Ratios must sum to 1.0 (current: {total_ratio})")
        return
    
    if not input_file.exists():
        console.print(f"❌ Input file not found: {input_file}")
        return
    
    console.print(f"Splitting: [cyan]{input_file}[/cyan]")
    console.print(f"Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    try:
        # Load data
        df = pd.read_csv(input_file)
        total_rows = len(df)
        
        console.print(f"📊 Total rows: {total_rows}")
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            console.print("🔀 Data shuffled")
        
        # Calculate split sizes
        train_size = int(total_rows * train_ratio)
        val_size = int(total_rows * val_ratio)
        test_size = total_rows - train_size - val_size  # Remaining rows
        
        # Split data
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_path = output_dir / "train.csv"
        val_path = output_dir / "validation.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Results summary
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Split", style="cyan")
        summary_table.add_column("Rows", style="green")
        summary_table.add_column("Percentage", style="yellow")
        summary_table.add_column("File", style="blue")
        
        summary_table.add_row("Train", str(len(train_df)), f"{len(train_df)/total_rows:.1%}", str(train_path))
        summary_table.add_row("Validation", str(len(val_df)), f"{len(val_df)/total_rows:.1%}", str(val_path))
        summary_table.add_row("Test", str(len(test_df)), f"{len(test_df)/total_rows:.1%}", str(test_path))
        
        console.print("\n📁 Split Results:")
        console.print(summary_table)
        
        console.print(f"\n✅ [bold green]Data split completed![/bold green]")
        console.print(f"📁 Output directory: [cyan]{output_dir}[/cyan]")
        
    except Exception as e:
        console.print(f"❌ Error splitting data: {e}")

@data.command()
@click.argument("source_files", nargs=-1, type=Path)
@click.option("--output", type=Path, default="data/processed/merged.csv", help="Output file")
@click.option("--deduplicate", is_flag=True, help="Remove duplicate entries")
def merge(source_files, output, deduplicate):
    """Merge multiple parallel data files."""
    console.print("🔗 [bold blue]Data Merging[/bold blue]\n")
    
    if not source_files:
        console.print("❌ No source files specified")
        return
    
    merged_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Merging files...", total=len(source_files))
        
        for file_path in source_files:
            progress.update(task, description=f"Loading {file_path.name}...")
            
            if not file_path.exists():
                console.print(f"⚠️  File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                merged_data.append(df)
                console.print(f"✅ Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                console.print(f"❌ Error loading {file_path}: {e}")
            
            progress.advance(task)
    
    if not merged_data:
        console.print("❌ No data loaded from source files")
        return
    
    # Merge all dataframes
    console.print("🔗 Combining data...")
    merged_df = pd.concat(merged_data, ignore_index=True)
    
    initial_count = len(merged_df)
    console.print(f"📊 Total rows after merge: {initial_count}")
    
    # Deduplicate if requested
    if deduplicate:
        console.print("🔍 Removing duplicates...")
        merged_df = merged_df.drop_duplicates(subset=["english", "toaripi"])
        final_count = len(merged_df)
        removed = initial_count - final_count
        console.print(f"🗑️  Removed {removed} duplicate rows")
    
    # Save merged data
    output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output, index=False)
    
    console.print(f"\n✅ [bold green]Data merging completed![/bold green]")
    console.print(f"📁 Output file: [cyan]{output}[/cyan]")
    console.print(f"📊 Final row count: {len(merged_df)}")

@data.command()
@click.argument("data_file", type=Path)
@click.option("--sample-size", type=int, default=10, help="Number of samples to show")
def preview(data_file, sample_size):
    """Preview parallel data with statistics."""
    console.print("👀 [bold blue]Data Preview[/bold blue]\n")
    
    if not data_file.exists():
        console.print(f"❌ File not found: {data_file}")
        return
    
    try:
        df = pd.read_csv(data_file)
        
        # Basic info
        info_panel = Panel(
            f"""
            [bold cyan]File Information[/bold cyan]
            
            File: {data_file}
            Rows: {len(df):,}
            Columns: {len(df.columns)}
            Size: {data_file.stat().st_size / 1024:.1f} KB
            """,
            title="📄 File Info",
            border_style="blue"
        )
        console.print(info_panel)
        
        # Column info
        console.print("\n📋 Columns:")
        columns_table = Table(show_header=True, header_style="bold magenta")
        columns_table.add_column("Column", style="cyan")
        columns_table.add_column("Type", style="green")
        columns_table.add_column("Non-null", style="yellow")
        columns_table.add_column("Sample Value", style="dim")
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = f"{df[col].count()}/{len(df)}"
            sample_val = str(df[col].iloc[0])[:50] + "..." if len(str(df[col].iloc[0])) > 50 else str(df[col].iloc[0])
            columns_table.add_row(col, col_type, non_null, sample_val)
        
        console.print(columns_table)
        
        # Sample data
        console.print(f"\n📋 Sample Data (first {sample_size} rows):")
        
        sample_df = df.head(sample_size)
        
        # Create preview table
        preview_table = Table(show_header=True, header_style="bold magenta", max_width=120)
        for col in df.columns:
            preview_table.add_column(col, style="cyan", max_width=30)
        
        for _, row in sample_df.iterrows():
            row_data = []
            for col in df.columns:
                val = str(row[col])
                if len(val) > 27:
                    val = val[:27] + "..."
                row_data.append(val)
            preview_table.add_row(*row_data)
        
        console.print(preview_table)
        
        # Statistics for text columns
        if "english" in df.columns and "toaripi" in df.columns:
            stats_table = Table(show_header=True, header_style="bold magenta")
            stats_table.add_column("Statistic", style="cyan")
            stats_table.add_column("English", style="green")
            stats_table.add_column("Toaripi", style="yellow")
            
            eng_lengths = df["english"].str.len()
            toar_lengths = df["toaripi"].str.len()
            
            stats_table.add_row("Average Length", f"{eng_lengths.mean():.1f}", f"{toar_lengths.mean():.1f}")
            stats_table.add_row("Min Length", f"{eng_lengths.min()}", f"{toar_lengths.min()}")
            stats_table.add_row("Max Length", f"{eng_lengths.max()}", f"{toar_lengths.max()}")
            stats_table.add_row("Median Length", f"{eng_lengths.median():.1f}", f"{toar_lengths.median():.1f}")
            
            console.print(f"\n📊 Text Statistics:")
            console.print(stats_table)
        
    except Exception as e:
        console.print(f"❌ Error reading file: {e}")

def save_validation_report(results: Dict[str, Any], report_path: Path):
    """Save validation results to a JSON report file."""
    import json
    from datetime import datetime
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "validation_results": results
    }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)