"""Tests for consolidated project README."""

from pathlib import Path


README_PATH = Path('README.md')
ASSET_BASE = Path('.')
REQUIRED_README_ASSETS = [
    'banner-3.png',
    'confusion_matrix-9.png',
    'roc_curve-15.png',
    'feature_distribution-5.png',
    'feature_importance-2.png',
    'Titanic Survival prediction final.pdf',
]


def test_readme_contains_required_sections():
    """README should include all required consolidated sections."""
    content = README_PATH.read_text(encoding='utf-8')
    required_headings = [
        '## Problem Statement',
        '## Project Overview',
        '## Business Use Case',
        '## Results at a Glance / Model Performance',
        '## Results and Key Findings',
        '## Visual Results',
        '## Tech Stack',
        '## Project Structure',
        '## How to Run',
        '## ML Pipeline - Step by Step',
        '## Selected Features Used',
        '## Key Learnings / Findings',
        '## Full Report',
        '## Connect',
        '## License',
    ]
    for heading in required_headings:
        assert heading in content


def test_readme_visual_assets_exist():
    """README visualization assets should resolve to existing files in repo root."""
    for asset in REQUIRED_README_ASSETS:
        assert (ASSET_BASE / asset).is_file(), f"Asset {asset} not found"
