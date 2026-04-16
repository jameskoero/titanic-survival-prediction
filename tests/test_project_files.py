from pathlib import Path


def test_requirements_file_exists():
    assert Path('requirements.txt').is_file()


def test_environment_file_exists():
    assert Path('environment.yml').is_file()
