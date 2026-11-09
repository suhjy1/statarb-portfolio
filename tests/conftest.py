import importlib.util

# the neural policies need the optional torch extra, install with pip install -e ".[neural]"
collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_neural_*.py"]
