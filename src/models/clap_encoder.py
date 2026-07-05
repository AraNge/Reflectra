import os
from typing import List, Union, Dict
import librosa
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor
from pathlib import Path


class PretrainedCLAPEncoder(torch.nn.Module):

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: Union[str, None] = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)

        self.model.to(self.device_name)

        if freeze:
            self.freeze()

        self.model.eval()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}

        text_features = self.model.get_text_features(**inputs)

        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output

        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features

    @torch.no_grad()
    def encode_audio(
        self,
        audio_paths: Union[str, List[str]],
        normalize: bool = True,
        target_sr: int = 48000,
        mono: bool = True,
    ) -> torch.Tensor:
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        audios = []

        for path in audio_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

            waveform, sample_rate = librosa.load(path, sr=target_sr, mono=mono)
            audios.append(waveform)

        inputs = self.processor(
            audio=audios,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}

        audio_features = self.model.get_audio_features(**inputs)

        if hasattr(audio_features, "pooler_output"):
            audio_features = audio_features.pooler_output

        if normalize:
            audio_features = F.normalize(audio_features, p=2, dim=-1)

        return audio_features

    @torch.no_grad()
    def similarity(
        self,
        texts: Union[str, List[str]],
        audio_paths: Union[str, List[str]],
    ) -> torch.Tensor:
        text_emb = self.encode_text(texts, normalize=True)
        audio_emb = self.encode_audio(audio_paths, normalize=True)

        return text_emb @ audio_emb.T

    def forward(self, batch: Dict[str, Union[List[str], List[str]]]):
        texts = batch["texts"]
        audio_paths = batch["audio_paths"]

        text_embeds = self.encode_text(texts, normalize=True)
        audio_embeds = self.encode_audio(audio_paths, normalize=True)

        logits = text_embeds @ audio_embeds.T

        return {
            "text_embeds": text_embeds,
            "audio_embeds": audio_embeds,
            "logits": logits,
        }


if __name__ == "__main__":
    encoder = PretrainedCLAPEncoder(freeze=True)

    query = "happy energetic pop song with female vocals and dance rhythm"

    audio_dir = Path("data/musiccaps_audio")
    audio_files = sorted(str(path) for path in audio_dir.glob("*.wav"))

    print(f"Found {len(audio_files)} wav files")
    print(audio_files[:5])

    scores = encoder.similarity(query, audio_files)

    print("Similarity scores:")
    print(scores.cpu().numpy())