Collection of Computer Vision tools
===================

A collection of computer vision tools and utilities.


Installation
===================

```console
# Install UV following instructions here: https://github.com/astral-sh/uv
# For Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12:
uv python install 3.12

# To view available versions:
uv python list

# Create a virtual environment using it
uv venv .venv-deploy -p 3.12

# activate an environment
source .venv-deploy/bin/activate

# install computert_vision_tools:
# clone computert_vision_tools
cd computert_vision_tools
uv pip install -e .

# installing additional packages in currently active environment:
uv add --active <PACKAGE_NAME>

# Managing an environment.

# to save an environment:
uv pip freeze > all-deps.txt
# manually remove packages installed from local repos.
# Save the pruned list as:
mv all-deps.txt deploy-requirements.in
# generate lockfile from your new input:
uv pip compile -o deploy-requirements.lock deploy-requirements.in

# to see the list of packages:
uv pip list

# To delete venv:
rm -rf .venv
```









