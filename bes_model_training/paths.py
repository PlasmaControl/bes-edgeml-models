from pathlib import Path


package_dir = Path(__file__).parent

repo_dir = package_dir.parent

data_dir = repo_dir / 'data'
data_dir.mkdir(exist_ok=True)

model_dir = repo_dir / 'models'
model_dir.mkdir(exist_ok=True)
