import os
from pathlib import Path
from backend import preprocessing
from backend.dl_model import generate_explainability_cams


def test_generate_explainers_smoke():
    img = Path("../runtime/uploads/test_fundus_tmp.png").resolve()
    assert img.exists(), "Test sample image not found"
    img_bytes = img.read_bytes()
    input_tensor, cam_base_rgb = preprocessing.preprocess_for_inference(img_bytes)
    outdir = Path("../runtime/cams/test_explainers")
    outdir.mkdir(parents=True, exist_ok=True)
    res = generate_explainability_cams(input_tensor, cam_base_rgb, 0, outdir / "test_sample")
    # At least combined_path or one of the individual maps should be created
    assert any(res.get(k) for k in ("combined_path", "attention_path", "gradcam_path")), "No explainer outputs created"
