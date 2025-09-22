"""
Data management commands for the Toaripi SLM CLI.
Handles dataset preparation, validation, and cultural appropriateness checks.
"""

import json
import click
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from ..models.enums import ContentType, AgeGroup, DataFormat
from ..models.dataset import Dataset
from ..data.preprocessing import ToaripiPreprocessor
from ..utils.helpers import get_file_hash

console = Console()


@click.group()
def data():
    """Data management commands for Toaripi educational content."""
    pass


@data.command()
@click.option('--input', '-i', 'input_path', required=True, type=Path,
              help='Input data file path')
@click.option('--output', '-o', 'output_path', type=Path,
              help='Output directory (default: data/processed/)')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'json', 'parquet']),
              default='csv', help='Output format')
@click.option('--content-types', multiple=True,
              type=click.Choice([ct.value for ct in ContentType]),
              default=['story'], help='Content types to prepare')
@click.option('--age-groups', multiple=True,
              type=click.Choice([ag.value for ag in AgeGroup]),
              default=['primary_lower'], help='Target age groups')
@click.option('--min-length', default=10, help='Minimum text length')
@click.option('--max-length', default=500, help='Maximum text length')
@click.option('--validate-cultural', is_flag=True, default=True,
              help='Enable cultural appropriateness validation')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
def prepare(input_path: Path, output_path: Optional[Path], output_format: str,
           content_types: List[str], age_groups: List[str], min_length: int,
           max_length: int, validate_cultural: bool, dry_run: bool):
    """Prepare and process Toaripi parallel data for training."""
    
    console.print("\n🔧 [bold blue]Preparing Toaripi Educational Dataset[/bold blue]\n")
    
    # Validate input file
    if not input_path.exists():
        console.print(f"❌ [red]Input file not found: {input_path}[/red]")
        raise click.Abort()
    
    # Set default output path
    if not output_path:
        output_path = Path("data/processed")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert string enums back to enum types
    content_type_enums = [ContentType(ct) for ct in content_types]
    age_group_enums = [AgeGroup(ag) for ag in age_groups]
    
    if dry_run:
        console.print("🧪 [yellow]DRY RUN MODE - No files will be modified[/yellow]\n")
    
    # Display configuration
    config_table = Table(title="Processing Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Input File", str(input_path))
    config_table.add_row("Output Directory", str(output_path))
    config_table.add_row("Output Format", output_format.upper())
    config_table.add_row("Content Types", ", ".join(content_types))
    config_table.add_row("Age Groups", ", ".join(age_groups))
    config_table.add_row("Text Length Range", f"{min_length}-{max_length} chars")
    config_table.add_row("Cultural Validation", "✓ Enabled" if validate_cultural else "✗ Disabled")
    
    console.print(config_table)
    console.print()
    
    if not dry_run:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize preprocessor
            task1 = progress.add_task("Initializing preprocessor...", total=1)
            preprocessor = ToaripiPreprocessor(
                min_length=min_length,
                max_length=max_length,
                validate_cultural=validate_cultural
            )
            progress.update(task1, completed=1)
            
            # Load and validate data
            task2 = progress.add_task("Loading and validating data...", total=1)
            try:
                dataset = preprocessor.load_parallel_data(input_path)
                progress.update(task2, completed=1)
                
                console.print(f"✓ [green]Loaded {len(dataset.parallel_texts)} text pairs[/green]")
                
            except Exception as e:
                progress.update(task2, completed=1)
                console.print(f"❌ [red]Failed to load data: {e}[/red]")
                raise click.Abort()
            
            # Process for each content type and age group
            task3 = progress.add_task("Processing educational content...", total=len(content_type_enums) * len(age_group_enums))
            
            results = {}
            for content_type in content_type_enums:
                for age_group in age_group_enums:
                    try:
                        processed_dataset = preprocessor.prepare_educational_dataset(
                            dataset, content_type, age_group
                        )
                        
                        # Generate output filename
                        filename = f"toaripi_{content_type.value}_{age_group.value}.{output_format}"
                        output_file = output_path / filename
                        
                        # Save processed data
                        if output_format == 'csv':
                            processed_dataset.to_csv(output_file)
                        elif output_format == 'json':
                            processed_dataset.to_json(output_file)
                        elif output_format == 'parquet':
                            processed_dataset.to_parquet(output_file)
                        
                        results[f"{content_type.value}_{age_group.value}"] = {
                            "file": str(output_file),
                            "records": len(processed_dataset.parallel_texts),
                            "content_type": content_type.value,
                            "age_group": age_group.value
                        }
                        
                        progress.update(task3, advance=1)
                        
                    except Exception as e:
                        console.print(f"❌ [red]Failed to process {content_type.value}/{age_group.value}: {e}[/red]")
                        progress.update(task3, advance=1)
                        continue
        
        # Display results
        console.print("\n📊 [bold green]Processing Results[/bold green]\n")
        
        results_table = Table()
        results_table.add_column("Dataset", style="cyan")
        results_table.add_column("Records", justify="right", style="yellow")
        results_table.add_column("Output File", style="white")
        
        for key, result in results.items():
            results_table.add_row(
                f"{result['content_type']}/{result['age_group']}",
                str(result['records']),
                result['file']
            )
        
        console.print(results_table)
        
        # Save processing manifest
        manifest_file = output_path / "processing_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump({
                "input_file": str(input_path),
                "processing_config": {
                    "content_types": content_types,
                    "age_groups": age_groups,
                    "min_length": min_length,
                    "max_length": max_length,
                    "cultural_validation": validate_cultural,
                    "output_format": output_format
                },
                "results": results,
                "timestamp": str(Path().cwd())  # placeholder for timestamp
            }, indent=2)
        
        console.print(f"\n✓ [green]Processing manifest saved: {manifest_file}[/green]")
    
    else:
        console.print("🧪 [yellow]Dry run completed - no files were modified[/yellow]")


@data.command()
@click.option('--file', '-f', 'file_path', required=True, type=Path,
              help='Data file to validate')
@click.option('--check-cultural', is_flag=True, default=True,
              help='Check cultural appropriateness')
@click.option('--check-educational', is_flag=True, default=True,
              help='Check educational suitability')
@click.option('--min-length', default=10, help='Minimum text length')
@click.option('--max-length', default=500, help='Maximum text length')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed validation results')
def validate(file_path: Path, check_cultural: bool, check_educational: bool,
            min_length: int, max_length: int, verbose: bool):
    """Validate data quality and cultural appropriateness."""
    
    console.print("\n🔍 [bold blue]Validating Toaripi Educational Data[/bold blue]\n")
    
    if not file_path.exists():
        console.print(f"❌ [red]File not found: {file_path}[/red]")
        raise click.Abort()
    
    # Display file info
    file_size = file_path.stat().st_size / 1024  # KB
    file_hash = get_file_hash(file_path)
    
    info_table = Table(title="File Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("File Path", str(file_path))
    info_table.add_row("File Size", f"{file_size:.2f} KB")
    info_table.add_row("File Hash", file_hash[:16] + "...")
    
    console.print(info_table)
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Initialize validator
        task1 = progress.add_task("Initializing validator...", total=1)
        preprocessor = ToaripiPreprocessor(
            min_length=min_length,
            max_length=max_length,
            validate_cultural=check_cultural
        )
        progress.update(task1, completed=1)
        
        # Load data
        task2 = progress.add_task("Loading data...", total=1)
        try:
            dataset = preprocessor.load_parallel_data(file_path)
            progress.update(task2, completed=1)
            console.print(f"✓ [green]Loaded {len(dataset.parallel_texts)} text pairs[/green]")
        except Exception as e:
            progress.update(task2, completed=1)
            console.print(f"❌ [red]Failed to load data: {e}[/red]")
            raise click.Abort()
        
        # Validate data quality
        task3 = progress.add_task("Validating data quality...", total=1)
        validation_results = preprocessor.validate_data_quality(dataset)
        progress.update(task3, completed=1)
        
        # Cultural appropriateness check
        if check_cultural:
            task4 = progress.add_task("Checking cultural appropriateness...", total=1)
            cultural_results = preprocessor.check_cultural_appropriateness(dataset)
            progress.update(task4, completed=1)
        else:
            cultural_results = {"passed": True, "issues": []}
        
        # Educational suitability check
        if check_educational:
            task5 = progress.add_task("Checking educational suitability...", total=1)
            educational_results = preprocessor.check_educational_suitability(dataset)
            progress.update(task5, completed=1)
        else:
            educational_results = {"passed": True, "issues": []}
    
    # Display validation results
    console.print("\n📋 [bold green]Validation Results[/bold green]\n")
    
    results_table = Table()
    results_table.add_column("Validation Type", style="cyan")
    results_table.add_column("Status", style="white")
    results_table.add_column("Issues Found", justify="right", style="yellow")
    
    status_quality = "✓ PASS" if validation_results.get("is_valid", False) else "❌ FAIL"
    status_cultural = "✓ PASS" if cultural_results.get("passed", False) else "❌ FAIL"
    status_educational = "✓ PASS" if educational_results.get("passed", False) else "❌ FAIL"
    
    results_table.add_row("Data Quality", status_quality, str(len(validation_results.get("issues", []))))
    if check_cultural:
        results_table.add_row("Cultural Appropriateness", status_cultural, str(len(cultural_results.get("issues", []))))
    if check_educational:
        results_table.add_row("Educational Suitability", status_educational, str(len(educational_results.get("issues", []))))
    
    console.print(results_table)
    
    # Show detailed issues if requested
    if verbose:
        all_issues = []
        all_issues.extend(validation_results.get("issues", []))
        all_issues.extend(cultural_results.get("issues", []))
        all_issues.extend(educational_results.get("issues", []))
        
        if all_issues:
            console.print("\n⚠️  [bold yellow]Detailed Issues[/bold yellow]\n")
            for i, issue in enumerate(all_issues[:10], 1):  # Show max 10 issues
                console.print(f"{i}. {issue}")
            
            if len(all_issues) > 10:
                console.print(f"\n... and {len(all_issues) - 10} more issues")
    
    # Overall status
    overall_pass = (validation_results.get("is_valid", False) and 
                   cultural_results.get("passed", False) and 
                   educational_results.get("passed", False))
    
    if overall_pass:
        console.print("\n✅ [bold green]Overall: VALIDATION PASSED[/bold green]")
    else:
        console.print("\n❌ [bold red]Overall: VALIDATION FAILED[/bold red]")


@data.command()
@click.option('--directory', '-d', type=Path, default=Path("data"),
              help='Directory to scan for datasets')
@click.option('--show-details', is_flag=True, help='Show detailed dataset information')
def list(directory: Path, show_details: bool):
    """List available datasets and their properties."""
    
    console.print("\n📁 [bold blue]Available Toaripi Datasets[/bold blue]\n")
    
    if not directory.exists():
        console.print(f"❌ [red]Directory not found: {directory}[/red]")
        raise click.Abort()
    
    # Find dataset files
    dataset_files = []
    for pattern in ["*.csv", "*.json", "*.parquet"]:
        dataset_files.extend(directory.rglob(pattern))
    
    if not dataset_files:
        console.print(f"No dataset files found in {directory}")
        return
    
    # Create datasets table
    table = Table()
    table.add_column("Dataset", style="cyan")
    table.add_column("Format", style="yellow")
    table.add_column("Size", justify="right", style="white")
    table.add_column("Location", style="dim white")
    
    if show_details:
        table.add_column("Records", justify="right", style="green")
        table.add_column("Modified", style="blue")
    
    for file_path in sorted(dataset_files):
        file_size = file_path.stat().st_size / 1024  # KB
        file_format = file_path.suffix[1:].upper()
        
        # Handle relative path calculation safely
        try:
            relative_path = file_path.relative_to(Path.cwd())
        except ValueError:
            # If the file is not in a subpath of current directory
            relative_path = file_path
        
        row = [
            file_path.name,
            file_format,
            f"{file_size:.1f} KB",
            str(relative_path.parent)
        ]
        
        if show_details:
            # Try to get record count
            try:
                if file_format.lower() == 'csv':
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    record_count = len(df)
                else:
                    record_count = "Unknown"
            except:
                record_count = "Error"
            
            modified_time = file_path.stat().st_mtime
            import datetime
            modified_str = datetime.datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
            
            row.extend([str(record_count), modified_str])
        
        table.add_row(*row)
    
    console.print(table)
    console.print(f"\nFound {len(dataset_files)} dataset files")


@data.command()
@click.option('--input', '-i', 'input_path', required=True, type=Path,
              help='Input file to convert')
@click.option('--output', '-o', 'output_path', type=Path,
              help='Output file path (auto-generated if not specified)')
@click.option('--from-format', type=click.Choice(['csv', 'json', 'parquet', 'usfm']),
              help='Source format (auto-detected if not specified)')
@click.option('--to-format', required=True, type=click.Choice(['csv', 'json', 'parquet']),
              help='Target format')
@click.option('--preserve-metadata', is_flag=True, default=True,
              help='Preserve metadata and structure')
def convert(input_path: Path, output_path: Optional[Path], from_format: Optional[str],
           to_format: str, preserve_metadata: bool):
    """Convert between different data formats."""
    
    console.print("\n🔄 [bold blue]Converting Dataset Format[/bold blue]\n")
    
    if not input_path.exists():
        console.print(f"❌ [red]Input file not found: {input_path}[/red]")
        raise click.Abort()
    
    # Auto-detect source format if not specified
    if not from_format:
        from_format = input_path.suffix[1:].lower()
        if from_format not in ['csv', 'json', 'parquet', 'usfm']:
            console.print(f"❌ [red]Cannot auto-detect format for: {input_path}[/red]")
            raise click.Abort()
    
    # Generate output path if not specified
    if not output_path:
        output_path = input_path.with_suffix(f'.{to_format}')
    
    console.print(f"Converting: {input_path} ({from_format.upper()}) → {output_path} ({to_format.upper()})")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Converting format...", total=3)
        
        try:
            # Load data
            progress.update(task, description="Loading source data...")
            if from_format == 'csv':
                import pandas as pd
                data = pd.read_csv(input_path)
            elif from_format == 'json':
                import pandas as pd
                data = pd.read_json(input_path)
            elif from_format == 'parquet':
                import pandas as pd
                data = pd.read_parquet(input_path)
            elif from_format == 'usfm':
                # Placeholder for USFM parsing
                console.print("❌ [red]USFM format not yet implemented[/red]")
                raise click.Abort()
            
            progress.update(task, advance=1)
            
            # Validate data structure
            progress.update(task, description="Validating data structure...")
            required_columns = ['english', 'toaripi']
            if not all(col in data.columns for col in required_columns):
                console.print(f"❌ [red]Missing required columns: {required_columns}[/red]")
                raise click.Abort()
            
            progress.update(task, advance=1)
            
            # Save in target format
            progress.update(task, description="Saving converted data...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if to_format == 'csv':
                data.to_csv(output_path, index=False)
            elif to_format == 'json':
                data.to_json(output_path, orient='records', indent=2)
            elif to_format == 'parquet':
                data.to_parquet(output_path, index=False)
            
            progress.update(task, advance=1)
            
        except Exception as e:
            console.print(f"❌ [red]Conversion failed: {e}[/red]")
            raise click.Abort()
    
    # Show results
    input_size = input_path.stat().st_size / 1024
    output_size = output_path.stat().st_size / 1024
    
    results_table = Table(title="Conversion Results")
    results_table.add_column("Property", style="cyan")
    results_table.add_column("Value", style="white")
    
    results_table.add_row("Input File", str(input_path))
    results_table.add_row("Output File", str(output_path))
    results_table.add_row("Records Converted", str(len(data)))
    results_table.add_row("Input Size", f"{input_size:.2f} KB")
    results_table.add_row("Output Size", f"{output_size:.2f} KB")
    results_table.add_row("Compression Ratio", f"{output_size/input_size:.2f}x")
    
    console.print("\n")
    console.print(results_table)
    console.print(f"\n✅ [bold green]Conversion completed successfully![/bold green]")


if __name__ == '__main__':
    data()