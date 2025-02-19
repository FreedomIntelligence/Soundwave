# Soundwave: *Less is More* for Speech-Text Alignment in LLMs

<div>
  <img src="assets/logo.png" style="width:160px; height:160px; margin-right:20px; float: left;"/>
  <div>
    <h2>✨ Highlights of Our Soundwave Model !️</h2>
    <ul>
      <font size="3"><li>A Speech-to-Text Model Bridging the Gap Between Speech and Text</li></font>
      <font size="3"><li>Utilizes Data-Efficient Strategy and Unique Architecture, Trained on Only 10k Hours of Data</li></font>
      <font size="3"><li>Exceptional Performance in Speech Translation and AIR-Bench Speech Tasks</li></font>
      <font size="3"><li>Retains Intelligence During Conversations, Ideal for Interactive Tasks</li></font>
    </ul>
  </div>
  <div style="clear:both;"></div>
</div>

<p align="center">
  <font size="3"><a href="https://huggingface.co/papers/2502.12900">🤗 HuggingFace</a>&nbsp｜&nbsp<a href="https://arxiv.org/abs/2502.12900">📃 Paper</a>｜&nbsp📼 Online Demo</a>&nbsp</font>
</p>

## 💌 News
> <ul>
>   <font size="3"><li>[19/02/2025] The online demo and model weights are coming soon. </li></font>
>   <font size="3"><li>[18/02/2025] Release the training and inference code. </li></font>
> </ul>

## Project Structure
```
.
├── assets/
│   ├── models/
│   │   ├── whisper/               # Whisper
│   │   └── Soundwave/             # Soundwave model weights
│   └── audio/                     # Directory for test audio files (e.g., .wav files)
├── README.md                      
├── run_inference.py               # Main inference script
└── Soundwave.py                   # Model architecture
```


## Getting Started

### Installation Requirements
<font size="3">Python version 3.10.11 is used in the Soundwave project.</font>
```bash
conda create -n soundwave python=3.10.11
conda activate soundwave
pip install -r requirements.txt 
```

## Inference
> <font size="3">Before starting, ensure you have at least 21GB of GPU memory to run our model inference.</font><br>

### Usage Command
<font size="3">To run the inference script and process the audio, use the following command:</font>
```bash
python run_inference.py --audio_tower <audio_tower_path> --base_model_path <base_model_path>
```

<font size="3">Options:
- `--audio_tower`: Path to the Whisper audio preprocessing model.
- `--base_model_path`: Path to the pre-trained Soundwave model.</font>
###
<font size="3">Below are some quick usage examples you can try:</font>
```python
import torch
import librosa
from run_inference import load_model, gen_model_inputs, CONFIG

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_model_path = "assets/models/Soundwave"
audio_tower_path = "assets/models/whisper"
model, audio_processor, tokenizer = load_model(base_model_path, audio_tower_path, device)

# apply chat template
prompt = "What does the person say?"
model_inputs = gen_model_inputs(tokenizer, prompt, device)

 # audio preprocess
audio_path = "assets/audio/example_1.wav"
audio, _ = librosa.load(audio_path, sr=CONFIG.sampling_rate, mono=True)
audio_feat = audio_processor(
    audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt"
).input_features.to(device, dtype=torch.float16)

 # inference
output_ids = model.generate(
    **model_inputs,
    audios=audio_feat,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.2,
)

input_token_len = model_inputs["input_ids"].shape[1]
response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

print(response)
```
## Citation
<font size="3">If you found this repository useful, please consider citing this work:</font>
```
@misc{zhang2025soundwavespeechtextalignmentllms,
      title={Soundwave: Less is More for Speech-Text Alignment in LLMs}, 
      author={Yuhao Zhang and Zhiheng Liu and Fan Bu and Ruiyu Zhang and Benyou Wang and Haizhou Li},
      year={2025},
      eprint={2502.12900},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12900}, 
}
```