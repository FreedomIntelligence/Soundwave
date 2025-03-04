import torch
import argparse
import librosa
from transformers import AutoTokenizer, WhisperProcessor
from Soundwave import SoundwaveForCausalLM


class BasicSetting:
    def __init__(self):
        self.sampling_rate = 16000
        self.audio_token_len = 1 
        self.stop = "</s>"
CONFIG = BasicSetting()


def load_model(model_path, device):
    # load model
    model = SoundwaveForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        torch_dtype=torch.float16,
        quantization_config=None,
        # attn_implementation="flash_attention_2"
    ).eval().to(device)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]
    model.config.llm_pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # load audio preprocessor
    audio_processor = WhisperProcessor.from_pretrained(model.config.audio_tower, torch_dtype=torch.float16)
    return model, audio_processor, tokenizer


def gen_model_inputs(tokenizer, prompt, device):
    system = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."
    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
    audio_placeholder = "\n"+audio_placeholder
    audio_placeholder_ids = tokenizer(audio_placeholder).input_ids

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    input_ids = []
    input_id = []

    system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system).input_ids + [eot_id]
    input_id += system

    user_input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(prompt).input_ids + [eot_id]
    assistant_input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens

    input_id += user_input_id
    input_id += assistant_input_id

    input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(device)
    attention_mask=input_ids.ne(tokenizer.pad_token_id)

    return dict(input_ids=input_ids, attention_mask=attention_mask)


def inference(model, audio_processor, tokenizer, prompt, audio_path, device):
    # apply chat template
    model_inputs = gen_model_inputs(tokenizer, prompt, device)

    # audio preprocess
    audio, _ = librosa.load(audio_path, sr=CONFIG.sampling_rate, mono=True)
    audio_feat = audio_processor(
        audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt"
    ).input_features.to(device, dtype=torch.float16)

    output_ids = model.generate(
        **model_inputs,
        audios=audio_feat,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.2,
    )

    input_ids = model_inputs["input_ids"]
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    outputs = outputs.strip()
    if outputs.endswith(CONFIG.stop):
        outputs = outputs[:-len(CONFIG.stop)]
    outputs = outputs.strip()
    
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_size', type=int, default=1280)
    parser.add_argument('--model_path', type=str, default="FreedomIntelligence/Soundwave")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.model_path
    
    model, audio_processor, tokenizer = load_model(model_path, device)

    prompt = "Please transcribe the following audio and then answer based on the audio's transcription."
    audio_path = "/mnt/nvme3n1/liuzhiheng/speech_copy/lzh/show_code/assets/audio/example_1.wav"

    response = inference(model, audio_processor, tokenizer, prompt, audio_path, device)

    print(f"{response}")