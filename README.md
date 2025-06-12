# DeepSeek R1 Implementation

## Motivation

This implementation heavily borrows the code from [Brendan Hogan's work](https://github.com/brendanhogan/DeepSeekRL-Extended) but restructured into a format optimized for learning and experimentation.

## Installation
```
uv sync
uv sync --extra dev # for linting and testing
uv pip install flash-attn --no-build-isolation
```

### Running the Experiment

1.  **Run Training:**
    ```bash
    python main.py \\
        --model_path Qwen/Qwen-VL-Chat \\
        --num_train_epochs 5 \\
        --eval_steps 50
        # Add other relevant arguments
    ```

