from src.models.clip_encoder import PretrainedCLIPEncoder
from src.models.clap_encoder import PretrainedCLAPEncoder
from tqdm import tqdm
import torch

def encode_in_batches_clip(
    model: PretrainedCLIPEncoder,
    image_paths: list[str],
    texts: list[str],
    batch_size: int,
):
    image_embeddings = []
    text_embeddings = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding CLIP"):
        end = start + batch_size

        batch_images = image_paths[start:end]
        image_emb = model.encode_image(batch_images)
        image_embeddings.append(image_emb.cpu())

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding CLIP text"):
        end = start + batch_size

        batch_texts = texts[start:end]
        text_emb = model.encode_text(batch_texts)
        text_embeddings.append(text_emb.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    return image_embeddings, text_embeddings



def encode_in_batches_clap(
    model: PretrainedCLAPEncoder,
    audio_paths: list[str],
    texts: list[str],
    batch_size: int,
):
    audio_embeddings = []
    text_embeddings = []

    for start in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding CLAP"):
        end = start + batch_size

        batch_audio_paths = audio_paths[start:end]
        audio_emb = model.encode_audio(batch_audio_paths)
        audio_embeddings.append(audio_emb.cpu())

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding CLAP text"):
        end = start + batch_size

        batch_texts = texts[start:end]
        text_emb = model.encode_text(batch_texts)
        text_embeddings.append(text_emb.cpu())

    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    return audio_embeddings, text_embeddings
