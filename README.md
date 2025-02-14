# Fine-Tuning Language and Multimodal models with LoRA

This repository contains notebooks for my experiments with fine-tuning Large Language and Multimodal Models using Parameter-Efficient Fine-Tuning (PEFT) techniques from Huggingface, specifically LoRA (Low-Rank Adaptation). 

## LLM Fine-tuning
The LLM notebook focuses on fine-tuning a model on the TriviaQA dataset.

#### Requirements

To run this notebook, you'll need the following:

*   **Python 3.10+** (Recommended to use a virtual environment)
*   **PyTorch:** Install the correct version for your operating system and hardware. Refer to the [PyTorch website](https://pytorch.org/) for installation instructions.  For Apple silicon (MPS), ensure you have a compatible version.
*   **Libraries:** Install the required libraries using `pip`:

    ```bash
    pip install transformers datasets peft accelerate bitsandbytes tqdm
    ```

    A `requirements.txt` file will be created later to manage this effectively.
*   **Jupyter Notebook:** Make sure you have Jupyter Notebook or a similar environment installed to run the `.ipynb` file.

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt # after creating the file
    ```
    To create the `requirements.txt` file use:
    ```bash
    pip freeze > requirements.txt
    ```

### Usage

1.  **Download the Notebook:** Download `main.ipynb` from the repository.

2.  **Open the Notebook:** Open `main.ipynb` in Jupyter Notebook.

3.  **Configure Training:**
    *   Modify the `TrainingConfig` class in the notebook to adjust training parameters such as:
        *   `MODEL_NAME`: The pre-trained LLM to use (e.g., `"Qwen/Qwen2.5-0.5B-Instruct"`).
        *   `BATCH_SIZE`: Adjust based on your GPU memory.
        *   `MAX_LENGTH`:  The maximum sequence length.
        *   `LEARNING_RATE`: The learning rate for training.
        *   `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`: LoRA-specific parameters.
    *   Change the `MODEL_NAME` if you want to use a different pre-trained model.

4.  **Run the Notebook:** Execute the cells sequentially.


## Troubleshooting

*   **CUDA Errors:** If you encounter CUDA-related errors, ensure you have a compatible PyTorch version and CUDA drivers installed correctly (if you're using an NVIDIA GPU).
*   **MPS Errors:** If using MPS, verify that PyTorch correctly detects your Apple silicon device.
*   **Memory Errors:** Memory errors typically indicate that the batch size is too large. Reduce `BATCH_SIZE` in the `TrainingConfig` or increase `GRADIENT_ACCUMULATION_STEPS` to compensate.
*   **Bits and Bytes warning "installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable."**: It's likely the installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable. Try:

    ```bash
    pip uninstall bitsandbytes
    pip install bitsandbytes --upgrade
    ```

    If you are using conda:

    ```bash
    conda uninstall bitsandbytes
    conda install -c conda-forge bitsandbytes
    ```

## Contributing

Contributions to this repository are welcome! Please submit a pull request with your changes.

## License

This project is licensed under the Apache License.