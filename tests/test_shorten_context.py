"""scripts/shorten_context.py: a Whisper cut to a shorter context is a Whisper built for it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def tiny():
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

    cfg = WhisperConfig(vocab_size=51865, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80,
                        max_source_positions=1500, max_target_positions=448)
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval()
    return model, WhisperFeatureExtractor(feature_size=80)


def test_cut_keeps_the_first_positions_and_takes_the_shorter_input():
    from shorten_context import shorten

    model, fe = tiny()
    before = model.model.encoder.embed_positions.weight.data[:400].clone()

    class Proc:  # the part of WhisperProcessor shorten() touches
        feature_extractor = fe

    proc = Proc()
    shorten(model, proc, 8)
    enc = model.model.encoder
    assert model.config.max_source_positions == 400 == enc.embed_positions.num_embeddings
    assert torch.equal(enc.embed_positions.weight.data, before)
    assert not enc.embed_positions.weight.requires_grad
    assert proc.feature_extractor.nb_max_frames == 800 and proc.feature_extractor.n_samples == 8 * 16000

    six_seconds = torch.zeros(96000).numpy()
    feats = proc.feature_extractor(six_seconds, sampling_rate=16000, return_tensors="pt").input_features
    assert feats.shape[-1] == 800
    with torch.no_grad():
        out = enc(feats)
    assert out.last_hidden_state.shape[1] == 400
    with pytest.raises(ValueError):
        enc(torch.zeros(1, 80, 3000))  # the 30 s input no longer fits
