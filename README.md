# <!--Repo Name-->

## Create Development Environment

### Activate and Deactivate Virtual Environment

```zsh
# Run the following command in the root directory of the project
$ uv sync
# Activate the virtual environment
$ source .venv/bin/activate

# Deactivate the virtual environment
$ deactivate
```

### Apply Formatter and Linter

```zsh
# Apply formatter
$ make fmt

# Apply linter
$ make lint
```

### Get Test and Coverage

```zsh
# test
$ make test

# Get coverage
$ make coverage

# Visualize coverage
$ make vis_coverage
```