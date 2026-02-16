from __future__ import annotations

import os
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import comfy.model_management as model_management
import folder_paths

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


# Load Qwen backend package from the official node checkout without importing AILab wrappers.
current_dir = Path(__file__).resolve().parent
qwen_pack_root = current_dir.parent / "ComfyUI-QwenTTS"
qwen_pkg_dir = qwen_pack_root / "qwen_tts"
if qwen_pack_root.exists() and str(qwen_pack_root) not in sys.path:
    sys.path.insert(0, str(qwen_pack_root))
if qwen_pkg_dir.exists() and str(qwen_pkg_dir) not in sys.path:
    sys.path.insert(0, str(qwen_pkg_dir))

_IMPORT_ERROR: Optional[str] = None
Qwen3TTSModel = None


LANGUAGE_CHOICES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]

LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

MODEL_ID_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]
_MODEL_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}


def _ensure_qwen_package() -> None:
    if not qwen_pkg_dir.exists():
        return

    pkg_name = "qwen_tts"
    if pkg_name not in sys.modules:
        try:
            import types

            module = types.ModuleType(pkg_name)
            module.__path__ = [str(qwen_pkg_dir)]
            sys.modules[pkg_name] = module
        except Exception:
            return

    core_dir = qwen_pkg_dir / "core"
    if not core_dir.exists() or "qwen_tts.core" in sys.modules:
        return

    try:
        import types
        import importlib.util
        from transformers import PreTrainedModel, PretrainedConfig

        core_module = types.ModuleType("qwen_tts.core")
        core_module.__path__ = [str(core_dir)]
        sys.modules["qwen_tts.core"] = core_module

        tokenizer_12hz_dir = core_dir / "tokenizer_12hz"
        if tokenizer_12hz_dir.exists():
            token_pkg = types.ModuleType("qwen_tts.core.tokenizer_12hz")
            token_pkg.__path__ = [str(tokenizer_12hz_dir)]
            sys.modules["qwen_tts.core.tokenizer_12hz"] = token_pkg

            cfg_path = tokenizer_12hz_dir / "configuration_qwen3_tts_tokenizer_v2.py"
            mdl_path = tokenizer_12hz_dir / "modeling_qwen3_tts_tokenizer_v2.py"

            if cfg_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
                    str(cfg_path),
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = mod
                    spec.loader.exec_module(mod)
                    core_module.Qwen3TTSTokenizerV2Config = getattr(mod, "Qwen3TTSTokenizerV2Config")

            if mdl_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
                    str(mdl_path),
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = mod
                    spec.loader.exec_module(mod)
                    core_module.Qwen3TTSTokenizerV2Model = getattr(mod, "Qwen3TTSTokenizerV2Model")

            class Qwen3TTSTokenizerV1Config(PretrainedConfig):
                model_type = "qwen3_tts_tokenizer_25hz"

            class Qwen3TTSTokenizerV1Model(PreTrainedModel):
                config_class = Qwen3TTSTokenizerV1Config

                def __init__(self, config):
                    super().__init__(config)

                def forward(self, *args, **kwargs):
                    raise RuntimeError("Tokenizer 25Hz is not supported in this node.")

            core_module.Qwen3TTSTokenizerV1Config = Qwen3TTSTokenizerV1Config
            core_module.Qwen3TTSTokenizerV1Model = Qwen3TTSTokenizerV1Model
    except Exception:
        return


def _load_qwen3_model():
    global Qwen3TTSModel, _IMPORT_ERROR
    if Qwen3TTSModel is not None:
        return Qwen3TTSModel

    _ensure_qwen_package()

    try:
        import importlib.util

        model_path = qwen_pkg_dir / "inference" / "qwen3_tts_model.py"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing qwen3_tts_model.py at {model_path}")

        spec = importlib.util.spec_from_file_location("qwen_tts.inference.qwen3_tts_model", str(model_path))
        if not spec or not spec.loader:
            raise ImportError("Failed to create spec for qwen3_tts_model")

        module = importlib.util.module_from_spec(spec)
        sys.modules["qwen_tts.inference.qwen3_tts_model"] = module
        spec.loader.exec_module(module)
        Qwen3TTSModel = getattr(module, "Qwen3TTSModel")
        return Qwen3TTSModel
    except Exception as e:
        _IMPORT_ERROR = str(e)
        print(f"[Custom-QwenTTS] Failed to import qwen3_tts_model: {e}")
        return None


def _available_devices() -> List[str]:
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def _sage_attn_available() -> bool:
    try:
        import sageattention  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_attention_impl(attention: str, device: str) -> Optional[str]:
    if not device.startswith("cuda"):
        return None

    if attention == "auto":
        if _flash_attn_available():
            return "flash_attention_2"
        if _sage_attn_available():
            return "sage_attn"
        return "sdpa"

    if attention == "sage_attn":
        return "sage_attn"

    if attention == "flash_attn":
        if _flash_attn_available():
            return "flash_attention_2"
        return None

    if attention in {"sdpa", "eager"}:
        return attention

    return None


def _patch_sage_attention(model: Any) -> None:
    try:
        from sageattention import sageattn
    except Exception:
        raise RuntimeError("sageattention is not installed. Please install it to use sage_attn.")

    patched = 0
    for name, module in model.model.named_modules():
        if getattr(module, "_qwen_sage_patched", False):
            continue
        if "attn" not in name.lower() and "Attention" not in type(module).__name__:
            continue
        if not hasattr(module, "forward"):
            continue

        original_forward = module.forward

        def make_sage_forward(orig_forward):
            def sage_forward(*args, **kwargs):
                if len(args) >= 3:
                    q, k, v = args[0], args[1], args[2]
                    if hasattr(q, "shape") and hasattr(k, "shape") and hasattr(v, "shape"):
                        attn_mask = kwargs.get("attention_mask", None)
                        try:
                            return sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                        except Exception:
                            return orig_forward(*args, **kwargs)
                return orig_forward(*args, **kwargs)

            return sage_forward

        module.forward = make_sage_forward(original_forward)
        module._qwen_sage_patched = True
        patched += 1

    print(f"[Custom-QwenTTS] sage_attn patched modules: {patched}")


def _resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return str(model_management.get_torch_device())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_choice == "cuda" and torch.cuda.is_available():
        return str(model_management.get_torch_device())

    if device_choice == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _resolve_dtype(precision: str, device: str):
    if device == "mps":
        if precision in ("bf16", "fp16"):
            return torch.float16
        return torch.float32

    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _maybe_autocast(device: str, precision: str):
    if not device.startswith("cuda"):
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _get_tts_paths() -> List[str]:
    folders = getattr(folder_paths, "folder_names_and_paths", {}) or {}
    for key in ("tts", "TTS"):
        if key in folders:
            return folder_paths.get_folder_paths(key) or []
    return []


def _model_store_root() -> str:
    tts_paths = _get_tts_paths()
    if tts_paths:
        return os.path.join(tts_paths[0], "Qwen3-TTS")
    return os.path.join(folder_paths.models_dir, "TTS", "Qwen3-TTS")


def _find_local_model(model_id: str) -> Optional[str]:
    model_name = model_id.split("/")[-1]
    candidates = []

    default_root = _model_store_root()
    if os.path.isdir(default_root):
        candidates.append(os.path.join(default_root, model_name))

    try:
        for root in _get_tts_paths():
            candidates.append(os.path.join(root, "Qwen3-TTS", model_name))
    except Exception:
        pass

    for path in candidates:
        if os.path.isdir(path) and os.listdir(path):
            return path
    return None


def _download_model(model_id: str) -> Optional[str]:
    if snapshot_download is None:
        return None

    target_root = _model_store_root()
    os.makedirs(target_root, exist_ok=True)
    target_dir = os.path.join(target_root, model_id.split("/")[-1])

    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return target_dir

    try:
        snapshot_download(repo_id=model_id, local_dir=target_dir, local_dir_use_symlinks=False)
        return target_dir
    except Exception as e:
        print(f"[Custom-QwenTTS] Download failed for {model_id}: {e}")
        return None


def _resolve_model_source(model_id: str) -> str:
    local_path = _find_local_model(model_id)
    if local_path:
        return local_path

    dl_path = _download_model(model_id)
    if dl_path:
        return dl_path

    return model_id


def _load_model(model_type: str, model_size: str, device_choice: str, precision: str, attention: str = "auto"):
    if Qwen3TTSModel is None:
        _load_qwen3_model()
    if Qwen3TTSModel is None:
        hint = _IMPORT_ERROR or "unknown import error"
        raise RuntimeError(
            "qwen_tts is not available. Please install dependencies in your ComfyUI environment "
            f"and check the package path. Import error: {hint}"
        )

    model_id = MODEL_ID_MAP.get((model_type, model_size))
    if model_id is None:
        raise ValueError(f"Unsupported model type/size: {model_type}/{model_size}")

    device = _resolve_device(device_choice)
    dtype = _resolve_dtype(precision, device)

    attn_impl = _resolve_attention_impl(attention, device)
    cache_key = (model_type, model_size, device, precision, attn_impl or "default")
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    source = _resolve_model_source(model_id)

    if device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    try:
        if attn_impl == "sage_attn":
            if not _sage_attn_available():
                print("[Custom-QwenTTS] sage_attn unavailable; fallback to default attention.")
                attn_impl = None
            else:
                model = Qwen3TTSModel.from_pretrained(source, device_map=device, dtype=dtype)
                _patch_sage_attention(model)
                print("[Custom-QwenTTS] attention: sage_attn")

        if attn_impl != "sage_attn":
            kwargs = {"device_map": device, "dtype": dtype}
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = Qwen3TTSModel.from_pretrained(source, **kwargs)
            if device.startswith("cuda") and attn_impl:
                print(f"[Custom-QwenTTS] attention: {attn_impl}")
    except Exception as e:
        if "attn_implementation" in str(e) or "flash" in str(e).lower():
            print("[Custom-QwenTTS] attention unavailable, fallback to default attention")
            model = Qwen3TTSModel.from_pretrained(source, device_map=device, dtype=dtype)
        else:
            raise e

    _MODEL_CACHE[cache_key] = model
    return model


def _set_seed(seed: int) -> None:
    if seed is None or seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))


def _get_prompt_item_class():
    try:
        _load_qwen3_model()
        module = __import__("qwen_tts.inference.qwen3_tts_model", fromlist=["VoiceClonePromptItem"])
        return getattr(module, "VoiceClonePromptItem", None)
    except Exception:
        return None


def _deserialize_prompt_item(data: Dict[str, Any]):
    cls = _get_prompt_item_class()
    if cls is None:
        return data
    return cls(
        ref_code=data.get("ref_code"),
        ref_spk_embedding=data.get("ref_spk_embedding"),
        x_vector_only_mode=bool(data.get("x_vector_only_mode")),
        icl_mode=bool(data.get("icl_mode")),
        ref_text=data.get("ref_text"),
    )


def _deserialize_prompt_items(items: List[Dict[str, Any]]) -> List[Any]:
    return [_deserialize_prompt_item(item) for item in items]


def _load_voice_from_file(path: str):
    if not path:
        return None

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if isinstance(data, dict) and "prompt" in data:
        payload = data["prompt"]
    else:
        payload = data

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return _deserialize_prompt_items(payload)
    return payload


def _audio_to_tuple(audio: Any) -> Tuple[np.ndarray, int]:
    waveform = None
    sr = None

    if isinstance(audio, dict):
        if "waveform" in audio:
            waveform = audio.get("waveform")
            sr = audio.get("sample_rate") or audio.get("sr") or audio.get("sampling_rate")
        elif "data" in audio and "sampling_rate" in audio:
            waveform = audio.get("data")
            sr = audio.get("sampling_rate")
        elif "audio" in audio and isinstance(audio["audio"], (tuple, list)):
            a0, a1 = audio["audio"]
            if isinstance(a0, (int, float)):
                sr, waveform = int(a0), a1
            else:
                waveform, sr = a0, int(a1)
    elif isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, (int, float)):
            sr, waveform = int(a0), a1
        else:
            waveform, sr = a0, int(a1)
    elif isinstance(audio, list) and len(audio) == 2:
        waveform, sr = audio[0], int(audio[1])

    if sr is None or waveform is None:
        raise ValueError("Invalid AUDIO input")

    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach()
        if wav.dim() > 1:
            wav = wav.squeeze()
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
        wav = wav.cpu().numpy()
    else:
        wav = np.asarray(waveform)

    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    wav = wav.astype(np.float32)
    if wav.size < 1024:
        wav = np.concatenate([wav, np.zeros(1024 - wav.size, dtype=np.float32)])

    return wav, int(sr)


def _to_comfy_audio(wavs: Any, sr: int) -> Dict[str, Any]:
    wav = wavs[0] if isinstance(wavs, list) and len(wavs) > 0 else wavs
    if isinstance(wav, np.ndarray):
        tensor = torch.from_numpy(wav).float()
    else:
        tensor = wav.float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": int(sr)}


def _model_supports_instruct(model: Any) -> bool:
    tts_model_size = str(getattr(getattr(model, "model", None), "tts_model_size", "")).lower()
    return tts_model_size not in {"0b6", "0.6b", "06b"}


def _generate_voice_clone_with_instruct(
    model: Any,
    text: Union[str, List[str]],
    language: Union[str, List[str], None] = None,
    instruct: Optional[Union[str, List[str]]] = None,
    ref_audio: Optional[Any] = None,
    ref_text: Optional[Union[str, List[Optional[str]]]] = None,
    x_vector_only_mode: Union[bool, List[bool]] = False,
    voice_clone_prompt: Optional[Any] = None,
    non_streaming_mode: bool = False,
    **kwargs,
) -> Tuple[List[Any], int]:
    texts = model._ensure_list(text)
    languages = model._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))

    if not _model_supports_instruct(model):
        instruct = None
    instructs = model._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

    if len(languages) == 1 and len(texts) > 1:
        languages = languages * len(texts)
    if len(instructs) == 1 and len(texts) > 1:
        instructs = instructs * len(texts)
    if not (len(texts) == len(languages) == len(instructs)):
        raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

    model._validate_languages(languages)

    if voice_clone_prompt is None:
        if ref_audio is None:
            raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
        prompt_items = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
        if len(prompt_items) == 1 and len(texts) > 1:
            prompt_items = prompt_items * len(texts)
        if len(prompt_items) != len(texts):
            raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
        voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
        ref_texts_for_ids = [it.ref_text for it in prompt_items]
    else:
        if isinstance(voice_clone_prompt, list):
            prompt_items = voice_clone_prompt
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            voice_clone_prompt_dict = voice_clone_prompt
            ref_texts_for_ids = None

    input_ids = model._tokenize_texts([model._build_assistant_text(t) for t in texts])

    ref_ids = None
    if ref_texts_for_ids is not None:
        ref_ids = []
        for rt in ref_texts_for_ids:
            if rt is None or rt == "":
                ref_ids.append(None)
            else:
                ref_ids.append(model._tokenize_texts([model._build_ref_text(rt)])[0])

    instruct_ids: List[Optional[torch.Tensor]] = []
    for ins in instructs:
        if ins is None or ins == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(model._tokenize_texts([model._build_instruct_text(ins)])[0])

    gen_kwargs = model._merge_generate_kwargs(**kwargs)

    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=languages,
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )

    codes_for_decode = []
    for i, codes in enumerate(talker_codes_list):
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        if ref_code_list is not None and ref_code_list[i] is not None:
            codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
        else:
            codes_for_decode.append(codes)

    wavs_all, fs = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

    wavs_out = []
    for i, wav in enumerate(wavs_all):
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        if ref_code_list is not None and ref_code_list[i] is not None:
            ref_len = int(ref_code_list[i].shape[0])
            total_len = int(codes_for_decode[i].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wavs_out.append(wav[cut:])
        else:
            wavs_out.append(wav)

    return wavs_out, fs


def _voice_clone_generate_with_instruct(
    reference_audio: Any,
    target_text: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    instruct: str = "",
    reference_text: str = "",
    x_vector_only: bool = False,
    voice: Any = None,
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not target_text or not target_text.strip():
        raise ValueError("target_text is required")

    prompt = None
    if voice is not None:
        if isinstance(voice, str) and voice.strip():
            prompt = _load_voice_from_file(voice.strip())
        elif isinstance(voice, list):
            prompt = voice

    if prompt is None:
        if (not reference_text or not reference_text.strip()) and not x_vector_only:
            raise ValueError("reference_text is required unless x_vector_only is enabled")
        if reference_audio is None:
            raise ValueError("reference_audio is required when voice is not provided")

    _set_seed(seed)
    model = _load_model("Base", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")

    with _maybe_autocast(_resolve_device(device), precision):
        if prompt is not None:
            wavs, sr = _generate_voice_clone_with_instruct(
                model=model,
                text=target_text,
                language=mapped_lang,
                instruct=instruct if instruct and instruct.strip() else None,
                voice_clone_prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        else:
            audio_tuple = _audio_to_tuple(reference_audio)
            wavs, sr = _generate_voice_clone_with_instruct(
                model=model,
                text=target_text,
                language=mapped_lang,
                instruct=instruct if instruct and instruct.strip() else None,
                ref_audio=audio_tuple,
                ref_text=reference_text.strip() if reference_text and reference_text.strip() else None,
                x_vector_only_mode=bool(x_vector_only),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        _MODEL_CACHE.clear()
        model_management.soft_empty_cache()
        try:
            import gc

            gc.collect()
            gc.collect()
        except Exception:
            pass
    return (audio,)


class UserQwen3TTSVoiceCloneInstructAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict(
                [
                    ("target_text", ("STRING", {"multiline": True, "default": "Hello, this is a cloned voice.", "tooltip": "Text to speak"})),
                    ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                    ("device", (_available_devices(), {"default": "auto"})),
                    ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                    ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
                ]
            ),
            "optional": OrderedDict(
                [
                    ("reference_audio", ("AUDIO", {"tooltip": "Reference audio for cloning (not needed if voice is provided)"})),
                    ("reference_text", ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of reference audio"})),
                    ("x_vector_only", ("BOOLEAN", {"default": False, "tooltip": "Skip ref_text by using speaker embedding only"})),
                    ("instruct", ("STRING", {"multiline": True, "default": "", "tooltip": "Optional speaking style/tone instruction"})),
                    ("voice", ("VOICE",)),
                    ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                    ("do_sample", ("BOOLEAN", {"default": False})),
                    ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                    ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                    ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                    ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                    ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                    ("unload_models", ("BOOLEAN", {"default": True})),
                    ("seed", ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF})),
                ]
            ),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Custom/QwenTTS"

    def generate(
        self,
        target_text,
        model_size,
        device,
        precision,
        language,
        reference_audio=None,
        reference_text="",
        x_vector_only=False,
        instruct="",
        voice=None,
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.9,
        repetition_penalty=1.0,
        attention="auto",
        unload_models=True,
        seed=-1,
    ):
        return _voice_clone_generate_with_instruct(
            reference_audio=reference_audio,
            target_text=target_text,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            instruct=instruct,
            reference_text=reference_text,
            x_vector_only=x_vector_only,
            voice=voice,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "User_Qwen3TTSVoiceCloneInstr_Advanced": UserQwen3TTSVoiceCloneInstructAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "User_Qwen3TTSVoiceCloneInstr_Advanced": "Voice Clone + Instruct (User QwenTTS) Advanced",
}
