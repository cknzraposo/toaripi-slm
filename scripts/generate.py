#!/usr/bin/env python3
"""
Content generation script for Toaripi SLM.

This script generates educational content using a fine-tuned Toaripi model.
"""

import click
import json
from pathlib import Path
from loguru import logger

from toaripi_slm import ToaripiGenerator, ContentType, AgeGroup, setup_logging


@click.command()
@click.option(
    '--model-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to fine-tuned Toaripi model'
)
@click.option(
    '--prompt',
    type=str,
    help='Text prompt for content generation'
)
@click.option(
    '--prompts-file',
    type=click.Path(exists=True, path_type=Path),
    help='JSON file containing multiple prompts'
)
@click.option(
    '--content-type',
    type=click.Choice(['story', 'vocabulary', 'qa', 'dialogue']),
    default='story',
    help='Type of content to generate'
)
@click.option(
    '--age-group',
    type=click.Choice(['primary', 'secondary', 'adult']),
    default='primary',
    help='Target age group for content'
)
@click.option(
    '--max-length',
    type=int,
    default=200,
    help='Maximum length of generated content'
)
@click.option(
    '--temperature',
    type=float,
    default=0.7,
    help='Temperature for text generation (0.0-2.0)'
)
@click.option(
    '--num-samples',
    type=int,
    default=1,
    help='Number of content samples to generate per prompt'
)
@click.option(
    '--output-file',
    type=click.Path(path_type=Path),
    help='Output file to save generated content (JSON format)'
)
@click.option(
    '--interactive',
    is_flag=True,
    help='Run in interactive mode'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--log-file',
    type=click.Path(path_type=Path),
    help='Optional log file path'
)
def generate(
    model_path: Path,
    prompt: str,
    prompts_file: Path,
    content_type: str,
    age_group: str,
    max_length: int,
    temperature: float,
    num_samples: int,
    output_file: Path,
    interactive: bool,
    log_level: str,
    log_file: Path
):
    """Generate educational content using Toaripi SLM."""
    
    # Setup logging
    setup_logging(level=log_level, log_file=str(log_file) if log_file else None)
    
    logger.info("Starting Toaripi content generation")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Content type: {content_type}")
    logger.info(f"Age group: {age_group}")
    
    try:
        # Load model
        logger.info("Loading Toaripi generator...")
        generator = ToaripiGenerator.load(str(model_path))
        logger.info("Generator loaded successfully")
        
        # Convert string enums to proper types
        content_type_enum = ContentType(content_type.upper())
        age_group_enum = AgeGroup(age_group.upper())
        
        results = []
        
        if interactive:
            # Interactive mode
            click.echo("\n🌺 Toaripi Educational Content Generator 🌺")
            click.echo("Type 'quit' or 'exit' to stop")
            click.echo("="*50)
            
            while True:
                try:
                    user_prompt = click.prompt("\nEnter your prompt")
                    
                    if user_prompt.lower() in ['quit', 'exit']:
                        break
                    
                    # Allow changing content type in interactive mode
                    content_choice = click.prompt(
                        "Content type (story/vocabulary/qa/dialogue)",
                        default=content_type,
                        show_default=True
                    )
                    
                    age_choice = click.prompt(
                        "Age group (primary/secondary/adult)",
                        default=age_group,
                        show_default=True
                    )
                    
                    try:
                        current_content_type = ContentType(content_choice.upper())
                        current_age_group = AgeGroup(age_choice.upper())
                    except ValueError as e:
                        click.echo(f"Invalid choice: {e}")
                        continue
                    
                    # Generate content
                    click.echo("\nGenerating content...")
                    
                    generated_content = generator.generate_content(
                        prompt=user_prompt,
                        content_type=current_content_type,
                        age_group=current_age_group,
                        max_length=max_length,
                        temperature=temperature
                    )
                    
                    # Display result
                    click.echo("\n" + "="*30)
                    click.echo(f"Generated {content_choice} for {age_choice} learners:")
                    click.echo("="*30)
                    click.echo(generated_content)
                    click.echo("="*30)
                    
                    # Store result
                    results.append({
                        'prompt': user_prompt,
                        'content_type': content_choice,
                        'age_group': age_choice,
                        'generated_content': generated_content,
                        'max_length': max_length,
                        'temperature': temperature
                    })
                    
                except KeyboardInterrupt:
                    click.echo("\nExiting...")
                    break
                except Exception as e:
                    click.echo(f"Error generating content: {e}")
        
        else:
            # Batch mode
            prompts_to_process = []
            
            if prompt:
                prompts_to_process.append(prompt)
            
            if prompts_file:
                logger.info(f"Loading prompts from: {prompts_file}")
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                
                if isinstance(prompt_data, list):
                    prompts_to_process.extend(prompt_data)
                elif isinstance(prompt_data, dict):
                    if 'prompts' in prompt_data:
                        prompts_to_process.extend(prompt_data['prompts'])
                    else:
                        prompts_to_process.append(prompt_data.get('prompt', ''))
            
            if not prompts_to_process:
                raise click.ClickException("No prompts provided. Use --prompt or --prompts-file")
            
            logger.info(f"Processing {len(prompts_to_process)} prompts")
            
            for i, current_prompt in enumerate(prompts_to_process, 1):
                logger.info(f"Generating content for prompt {i}/{len(prompts_to_process)}")
                
                for sample in range(num_samples):
                    try:
                        generated_content = generator.generate_content(
                            prompt=current_prompt,
                            content_type=content_type_enum,
                            age_group=age_group_enum,
                            max_length=max_length,
                            temperature=temperature
                        )
                        
                        result = {
                            'prompt': current_prompt,
                            'content_type': content_type,
                            'age_group': age_group,
                            'generated_content': generated_content,
                            'max_length': max_length,
                            'temperature': temperature,
                            'sample_number': sample + 1
                        }
                        
                        results.append(result)
                        
                        # Print to console
                        click.echo(f"\n--- Prompt {i}, Sample {sample + 1} ---")
                        click.echo(f"Prompt: {current_prompt}")
                        click.echo(f"Generated: {generated_content}")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate content for prompt {i}, sample {sample + 1}: {e}")
        
        # Save results if output file specified
        if output_file and results:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                'metadata': {
                    'model_path': str(model_path),
                    'content_type': content_type,
                    'age_group': age_group,
                    'max_length': max_length,
                    'temperature': temperature,
                    'num_samples': num_samples,
                    'total_generated': len(results)
                },
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(results)} generated samples to: {output_file}")
        
        logger.success(f"Content generation completed! Generated {len(results)} samples")
        
        if not interactive:
            # Print summary
            click.echo("\n" + "="*50)
            click.echo("GENERATION SUMMARY")
            click.echo("="*50)
            click.echo(f"Model: {model_path}")
            click.echo(f"Content type: {content_type}")
            click.echo(f"Age group: {age_group}")
            click.echo(f"Total samples: {len(results)}")
            if output_file:
                click.echo(f"Output file: {output_file}")
            click.echo("="*50)
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise click.ClickException(f"Content generation failed: {e}")


if __name__ == '__main__':
    generate()
if __name__ == "__main__":
    main()