# Soundwave: *Less is More* for Speech-Text Alignment in LLMs

<img src="logo.svg" align=center alt="Soundwave" style="width:200px; height:200px;" >

## Introduction to Soundwave Model

**Soundwave** is a novel speech-to-text model that addresses two key challenges in the interaction between speech and text: the representation space gap and sequence length inconsistency. Traditional end-to-end speech large language models (LLMs) rely on large-scale annotated data, but Soundwave utilizes a data-efficient training strategy and a unique architecture to achieve superior performance. Our results show that Soundwave outperforms advanced speech LLMs in tasks such as speech translation and AIR-Bench speech, using only a fraction of the training data. Additionally, Soundwave retains its intelligence during conversations, making it effective for interactive tasks.

## Project Directory Structure
```
.
├── assets/
│   ├── models/
│   │   ├── whisper/               # Whisper
│   │   └── Soundwave/             # Soundwave model weights
│   └── audio/                     # Directory for audio files (e.g., .wav files)
├── README.md                      # Project documentation
├── run_inference.py               # Main inference script
└── Soundwave.py                   # Model architecture
```


### Installation
The python version is 3.10.11, and the other requirements package can be installed with: ``` pip install -r requirements.txt ```
## Inference Function Parameters

The `inference` function in `run_inference.py` requires the following parameters:

- `model`: The loaded Soundwave model.
- `audio_processor`: The loaded Whisper audio processor.
- `tokenizer`: The loaded tokenizer instance for the model.
- `prompt`: The task prompt (instruction).
- `audio_path`: The file path to the audio that needs to be processed (e.g., `.wav` file).



### Example

```python
prompt = "Please transcribe the following audio and then answer based on the audio's transcription."
audio_path = "assets/audio/example_1.wav"
response = inference(model, audio_processor, tokenizer, prompt, audio_path)
```

## Usage Command
To run the inference script and process the audio, use the following command:

```bash
python run_inference.py --audio_tower <audio_tower_path> --base_model_path <base_model_path>
```
Options:
- `--audio_tower`: Path to the Whisper audio preprocessing model.
- `--base_model_path`: Path to the pre-trained Soundwave model.

