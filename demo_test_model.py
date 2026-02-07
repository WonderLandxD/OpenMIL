from slide_encoder_models import list_models, create_slide_encoder
import torch


DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PATCHES = 500
PATCH_SIZE_LV0 = 1024

FEAT_DIMS = {
    ########## Basic MIL models ##########
    "abmil": 1024,
    "gated_abmil": 1024,
    "transmil": 1024,
    "amdmil": 1024,
    "clam_sb": 1024,
    "clam_mb": 1024,
    "dsmil": 1024,
    "2dmamba": 1024,
    "retmil": 1024,
    "wikg": 1024,
    "ilramil": 1024,
    ########## Pre-trained models ##########
    "titan": 768,
    "prism": 2560,
    "gigapath": 1536,
    "madeleine": 512,
    "chief": 768,
    "feather_uni_v1": 1024,
    "feather_uni_v2": 1536,
    "feather_conch_v1_5": 768,
}

NEED_COORDS = {"2dmamba", "titan", "gigapath"}
NEED_PATCH_SIZE = {"titan"}


def make_coords(n_patches: int, patch_size: int, dtype: torch.dtype, device: str):
    grid_w = int(n_patches ** 0.5)
    grid_h = (n_patches + grid_w - 1) // grid_w
    coords = []
    for i in range(n_patches):
        row = i // grid_w
        col = i % grid_w
        coords.append([col * patch_size, row * patch_size])
    return torch.tensor(coords, dtype=torch.int64, device=device).unsqueeze(0)


models = list_models()
for model_name in models:
    print("=" * 60)
    print(f"Creating Model: {model_name}")
    run_dtype = torch.float16 if model_name == "gigapath" else DTYPE
    model = create_slide_encoder(model_name, num_classes=10).to(DEVICE).to(run_dtype)
    feat_dim = FEAT_DIMS[model_name]
    input_dict = {"feats": torch.randn(1, N_PATCHES, feat_dim, device=DEVICE, dtype=run_dtype)}
    if model_name in NEED_COORDS:
        input_dict["coords"] = make_coords(N_PATCHES, PATCH_SIZE_LV0, DTYPE, DEVICE)
    if model_name in NEED_PATCH_SIZE:
        input_dict["patch_size_lv0"] = PATCH_SIZE_LV0
    output = model(input_dict)
    print(f"Output shape: {output.shape}, min: {output.min()}, max: {output.max()}")
    print("=" * 60)
    print(f"Model: {model_name} is working ✓")
    print("=" * 60)