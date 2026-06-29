"""`memo multimodal` command group — multi-modal (image/audio) corpus.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(multimodal_group)`.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- multi-modal commands (gamechanger #17) ---------------------------------------


@click.group(name="multimodal")
def multimodal_group() -> None:
    """Multi-modal memory with universal embeddings."""
    pass


@multimodal_group.command(name="add-image")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--memory-id", "memory_id", help="Associated memory ID")
def multimodal_add_image(image_path: str, memory_id: str | None) -> None:
    """Add an image to the multi-modal corpus.

    Example: memo multimodal add-image /path/to/image.png --memory-id abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    content = mem.multimodal.add_image(Path(image_path), memory_id)

    console.print("[green]Image added[/green]")
    console.print(f"Content ID: {content.id}")
    console.print(f"Modality: {content.modality}")


@multimodal_group.command(name="add-audio")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--memory-id", "memory_id", help="Associated memory ID")
def multimodal_add_audio(audio_path: str, memory_id: str | None) -> None:
    """Add audio to the multi-modal corpus.

    Example: memo multimodal add-audio /path/to/audio.mp3 --memory-id abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    content = mem.multimodal.add_audio(Path(audio_path), memory_id)

    console.print("[green]Audio added[/green]")
    console.print(f"Content ID: {content.id}")
    console.print(f"Modality: {content.modality}")


@multimodal_group.command(name="search-images")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def multimodal_search_images(query: str, limit: int) -> None:
    """Search by text, find images.

    Example: memo multimodal search-images "architecture diagram"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.multimodal.search.search_text_find_images(query, limit=limit)

    console.print(f"[bold]Results: {len(results)} images[/bold]")
    for r in results:
        console.print(f"  {r.content_id[:8]} - similarity: {r.similarity:.2f}")


@multimodal_group.command(name="search-audio")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def multimodal_search_audio(query: str, limit: int) -> None:
    """Search by text, find audio.

    Example: memo multimodal search-audio "meeting notes"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.multimodal.search.search_text_find_audio(query, limit=limit)

    console.print(f"[bold]Results: {len(results)} audio[/bold]")
    for r in results:
        console.print(f"  {r.content_id[:8]} - similarity: {r.similarity:.2f}")


@multimodal_group.command(name="search-all")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Maximum results per modality")
def multimodal_search_all(query: str, limit: int) -> None:
    """Search across all modalities.

    Example: memo multimodal search-all "project documentation"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.multimodal.search.search_all_modalities(query, limit=limit)

    console.print("[bold]Results across modalities[/bold]")
    for modality, mod_results in results.items():
        console.print(f"\n[cyan]{modality}:[/cyan] {len(mod_results)} results")
        for r in mod_results[:5]:
            console.print(f"  {r.content_id[:8]} - similarity: {r.similarity:.2f}")
