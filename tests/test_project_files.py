"""Project file presence and baseline content tests."""

from pathlib import Path


def test_requirements_file_exists():
    """Ensure requirements.txt is present for pip-based workflows."""
    assert Path('requirements.txt').is_file()


def test_environment_file_exists():
    """Ensure environment.yml is present for conda-based workflows."""
    assert Path('environment.yml').is_file()


def test_requirements_has_core_dependencies():
    """Ensure core runtime dependencies are listed in requirements.txt."""
    content = Path('requirements.txt').read_text(encoding='utf-8')
    assert 'pandas' in content
    assert 'numpy' in content
    assert 'scikit-learn' in content


def test_environment_has_conda_dependencies():
    """Ensure conda environment file declares channels and dependencies."""
    content = Path('environment.yml').read_text(encoding='utf-8')
    assert 'channels:' in content
    assert 'dependencies:' in content
    assert 'python=' in content


def test_titanic_model_saves_visualizations_to_outputs_folder():
    """Ensure visual outputs are saved under outputs/ paths."""
    content = Path('titanic_model.py').read_text(encoding='utf-8')
    assert "plt.savefig('outputs/confusion_matrix.png'" in content
    assert "plt.savefig('outputs/roc_curve.png'" in content
    assert "plt.savefig('outputs/feature_distribution.png'" in content
    assert "plt.savefig('outputs/feature_importance.png'" in content
