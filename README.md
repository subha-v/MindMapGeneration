# MindMapGeneration

1. Download [Python](https://www.python.org) version >= 3.8.

1. Create a virtual environment.
    ```bash
    python3 -m venv .venv.nosync
    ```

1. Enter the virtual environment.
    ```
    source .venv.nosync/bin/activate
    ```
    You should see your bash prompt prefixed with `(.venv.nosync)` which
    indicates that you are in the virtual enivronment. All subsequent commands
    in this README should be run in the virtual environment. To leave the
    virtual environment, use `deactivate`.

1. Install dependencies:
    ```bash
    pip3 install hlda nltk sklearn
    ```

1. Run script:
    ```bash
    python3 top2vec_testing.py
    ```
