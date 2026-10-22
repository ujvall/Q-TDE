import os
import cv2
import torch
import numpy as np
import qrcode

from torch import nn
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import transforms


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


VAL_DIR = "data/val"

VIT_MODEL_DIR = "models/vit/saved"

AE_MODEL_PATH = (
    "models/autoencoder/saved/autoencoder.pth"
)


VERSION = 3
MODULE_SIZE = 10
SIZE = 37
SYMBOL_SIZE = 29

TOTAL_CODEWORDS = 70
ECC_CODEWORDS = 26
BLOCK_TOTAL = 70
BLOCK_ECC = 26


# ============================================================
# GF(256)
# ============================================================

EXP_TABLE = [0] * 512
LOG_TABLE = [0] * 256


def initialize_gf():

    value = 1

    for i in range(255):

        EXP_TABLE[i] = value
        LOG_TABLE[value] = i

        value <<= 1

        if value & 0x100:
            value ^= 0x11D

    for i in range(255, 512):

        EXP_TABLE[i] = EXP_TABLE[i - 255]


initialize_gf()


def gf_multiply(a, b):

    if a == 0 or b == 0:
        return 0

    return EXP_TABLE[
        LOG_TABLE[a] + LOG_TABLE[b]
    ]


# ============================================================
# QR MASK
# ============================================================

def mask_condition(mask, row, col):

    if mask == 0:
        return (row + col) % 2 == 0

    if mask == 1:
        return row % 2 == 0

    if mask == 2:
        return col % 3 == 0

    if mask == 3:
        return (row + col) % 3 == 0

    if mask == 4:
        return (row // 2 + col // 3) % 2 == 0

    if mask == 5:
        return (
            ((row * col) % 2)
            + ((row * col) % 3)
        ) == 0

    if mask == 6:
        return (
            (
                ((row * col) % 2)
                + ((row * col) % 3)
            ) % 2
        ) == 0

    if mask == 7:
        return (
            (
                ((row + col) % 2)
                + ((row * col) % 3)
            ) % 2
        ) == 0

    raise ValueError("Invalid QR mask")


# ============================================================
# REFERENCE QR
# ============================================================

def generate_reference(number):

    payload_index = number - 1

    qr = qrcode.QRCode(
        version=VERSION,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=MODULE_SIZE,
        border=4
    )

    qr.add_data(
        f"https://example.com/page/{payload_index}"
    )

    qr.make(fit=False)

    return np.array(
        qr.get_matrix(),
        dtype=bool
    )


# ============================================================
# MODULE SAMPLING
# ============================================================

def sample_modules(image):

    image = cv2.resize(
        image,
        (
            SIZE * MODULE_SIZE,
            SIZE * MODULE_SIZE
        ),
        interpolation=cv2.INTER_NEAREST
    )

    modules = np.zeros(
        (SIZE, SIZE),
        dtype=bool
    )

    for r in range(SIZE):

        for c in range(SIZE):

            y = (
                r * MODULE_SIZE
                + MODULE_SIZE // 2
            )

            x = (
                c * MODULE_SIZE
                + MODULE_SIZE // 2
            )

            modules[r, c] = (
                image[y, x] < 128
            )

    return modules


# ============================================================
# QUIET ZONE
# ============================================================

def remove_quiet_zone(matrix):

    return matrix[
        4:4 + SYMBOL_SIZE,
        4:4 + SYMBOL_SIZE
    ]


# ============================================================
# FORMAT INFORMATION
# ============================================================

def format_bits_from_matrix(matrix):

    coordinates = [
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (8, 7),
        (8, 8),
        (7, 8),
        (5, 8),
        (4, 8),
        (3, 8),
        (2, 8),
        (1, 8),
        (0, 8)
    ]

    bits = 0

    for r, c in coordinates:

        bits <<= 1

        if matrix[r, c]:
            bits |= 1

    return bits


def decode_mask_pattern(matrix):

    format_value = (
        format_bits_from_matrix(matrix)
    )

    format_data = (
        format_value ^ 0x5412
    )

    data = format_data >> 10

    mask = data & 0x07

    return mask


# ============================================================
# FUNCTIONAL MODULE MAP
# ============================================================

def build_functional_map():

    functional = np.zeros(
        (SYMBOL_SIZE, SYMBOL_SIZE),
        dtype=bool
    )

    finder_origins = [
        (0, 0),
        (0, SYMBOL_SIZE - 7),
        (SYMBOL_SIZE - 7, 0)
    ]

    for origin_r, origin_c in finder_origins:

        for r in range(
            origin_r - 1,
            origin_r + 8
        ):

            for c in range(
                origin_c - 1,
                origin_c + 8
            ):

                if (
                    0 <= r < SYMBOL_SIZE
                    and 0 <= c < SYMBOL_SIZE
                ):

                    functional[r, c] = True

    for c in range(
        8,
        SYMBOL_SIZE - 8
    ):

        functional[6, c] = True

    for r in range(
        8,
        SYMBOL_SIZE - 8
    ):

        functional[r, 6] = True

    alignment_center = 22

    for r in range(
        alignment_center - 2,
        alignment_center + 3
    ):

        for c in range(
            alignment_center - 2,
            alignment_center + 3
        ):

            functional[r, c] = True

    for c in range(9):
        functional[8, c] = True

    for r in range(9):
        functional[r, 8] = True

    for c in range(
        SYMBOL_SIZE - 8,
        SYMBOL_SIZE
    ):

        functional[8, c] = True

    for r in range(
        SYMBOL_SIZE - 7,
        SYMBOL_SIZE
    ):

        functional[r, 8] = True

    functional[
        SYMBOL_SIZE - 8,
        8
    ] = True

    return functional


# ============================================================
# DATA COORDINATES
# ============================================================

def get_data_coordinates():

    functional = build_functional_map()

    coordinates = []

    upward = True

    col = SYMBOL_SIZE - 1

    while col > 0:

        if col == 6:
            col -= 1

        if upward:

            rows = range(
                SYMBOL_SIZE - 1,
                -1,
                -1
            )

        else:

            rows = range(
                SYMBOL_SIZE
            )

        for row in rows:

            for current_col in (
                col,
                col - 1
            ):

                if not functional[
                    row,
                    current_col
                ]:

                    coordinates.append(
                        (row, current_col)
                    )

        upward = not upward
        col -= 2

    return coordinates


# ============================================================
# CODEWORD EXTRACTION
# ============================================================

def extract_codewords(matrix):

    symbol = remove_quiet_zone(
        matrix
    )

    mask = decode_mask_pattern(
        symbol
    )

    coordinates = get_data_coordinates()

    if len(coordinates) != 567:

        raise ValueError(
            f"Expected 567 data modules, "
            f"got {len(coordinates)}"
        )

    bits = []

    for row, col in coordinates:

        bit = symbol[row, col]

        if mask_condition(
            mask,
            row,
            col
        ):

            bit = not bit

        bits.append(
            1 if bit else 0
        )

    codewords = []

    for i in range(
        TOTAL_CODEWORDS
    ):

        value = 0

        for j in range(8):

            value <<= 1

            value |= bits[
                i * 8 + j
            ]

        codewords.append(value)

    return codewords


# ============================================================
# ECC SCORE
# ============================================================

def calculate_ecc_score(
    image,
    reference
):

    modules = sample_modules(
        image
    )

    module_difference = int(
        np.sum(
            modules != reference
        )
    )

    codewords = extract_codewords(
        modules
    )

    syndromes = []

    for syndrome_index in range(
        BLOCK_ECC
    ):

        result = 0

        for position, codeword in enumerate(
            codewords
        ):

            if codeword == 0:
                continue

            exponent = (
                syndrome_index
                * (
                    BLOCK_TOTAL
                    - 1
                    - position
                )
            ) % 255

            coefficient = EXP_TABLE[
                exponent
            ]

            result ^= gf_multiply(
                codeword,
                coefficient
            )

        syndromes.append(
            result
        )

    residual = sum(
        syndromes
    )

    nonzero = sum(
        value != 0
        for value in syndromes
    )

    return (
        module_difference,
        nonzero,
        residual
    )


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                3,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                3,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                3,
                2,
                1
            ),

            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                16,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                16,
                1,
                4,
                2,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.decoder(
            self.encoder(x)
        )


def calculate_ae_score(
    image,
    model
):

    image = cv2.resize(
        image,
        (128, 128)
    )

    blur = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    detail = cv2.absdiff(
        image,
        blur
    )

    detail = (
        detail.astype(
            np.float32
        ) / 255.0
    )

    tensor = torch.tensor(
        detail,
        dtype=torch.float32
    ).unsqueeze(
        0
    ).unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        reconstructed = model(
            tensor
        )

        error = torch.abs(
            tensor - reconstructed
        )[0, 0].cpu().numpy()

    return float(
        np.mean(error)
    )


# ============================================================
# VIT SCORE
# ============================================================

def calculate_vit_score(
    image,
    model,
    processor
):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    image_rgb = cv2.resize(
        image_rgb,
        (224, 224)
    )

    transform = transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])

    tensor = transform(
        image_rgb
    ).unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        outputs = model(
            pixel_values=tensor
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    return float(
        probabilities[0, 1]
        .cpu()
        .item()
    )


# ============================================================
# GET VALIDATION PAIRS
# ============================================================

def get_validation_numbers():

    genuine_dir = os.path.join(
        VAL_DIR,
        "genuine"
    )

    tampered_dir = os.path.join(
        VAL_DIR,
        "tampered"
    )

    numbers = []

    for filename in os.listdir(
        genuine_dir
    ):

        if not filename.startswith(
            "genuine_"
        ):

            continue

        if not filename.endswith(
            ".png"
        ):

            continue

        number_text = filename[
            len("genuine_"):
            -len(".png")
        ]

        try:

            number = int(
                number_text
            )

        except ValueError:

            continue

        tampered_filename = (
            f"tampered_{number:03d}.png"
        )

        if os.path.exists(
            os.path.join(
                tampered_dir,
                tampered_filename
            )
        ):

            numbers.append(
                number
            )

    return sorted(numbers)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "VALIDATION BRANCH EVALUATION"
    )
    print(
        "============================"
    )

    print()
    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # LOAD VIT
    # --------------------------------------------------------

    print()
    print(
        "Loading ViT..."
    )

    processor = (
        ViTImageProcessor.from_pretrained(
            VIT_MODEL_DIR
        )
    )

    vit_model = (
        ViTForImageClassification
        .from_pretrained(
            VIT_MODEL_DIR
        )
    )

    vit_model.to(device)
    vit_model.eval()

    print(
        "ViT loaded."
    )

    # --------------------------------------------------------
    # LOAD AUTOENCODER
    # --------------------------------------------------------

    print()
    print(
        "Loading Autoencoder..."
    )

    ae_model = Autoencoder().to(
        device
    )

    ae_model.load_state_dict(
        torch.load(
            AE_MODEL_PATH,
            map_location=device
        )
    )

    ae_model.eval()

    print(
        "Autoencoder loaded."
    )

    # --------------------------------------------------------
    # VALIDATION PAIRS
    # --------------------------------------------------------

    numbers = get_validation_numbers()

    print()
    print(
        "Validation pairs:",
        len(numbers)
    )

    print()
    print(
        "VALIDATION RESULTS"
    )

    print(
        "================================================================================================"
    )

    print(
        f"{'Pair':<8}"
        f"{'Label':<12}"
        f"{'ViT':<14}"
        f"{'AE':<14}"
        f"{'ECC_Mod':<12}"
        f"{'ECC_Syn':<12}"
        f"{'ECC_Residual':<15}"
    )

    print(
        "------------------------------------------------------------------------------------------------"
    )

    results = []

    for number in numbers:

        reference = generate_reference(
            number
        )

        for label in [
            "Genuine",
            "Tampered"
        ]:

            filename = (
                f"{label.lower()}_{number:03d}.png"
            )

            path = os.path.join(
                VAL_DIR,
                label.lower(),
                filename
            )

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:

                print(
                    "ERROR:",
                    path
                )

                continue

            vit_score = (
                calculate_vit_score(
                    image,
                    vit_model,
                    processor
                )
            )

            ae_score = (
                calculate_ae_score(
                    image,
                    ae_model
                )
            )

            ecc_mod, ecc_syn, ecc_res = (
                calculate_ecc_score(
                    image,
                    reference
                )
            )

            results.append({

                "pair": number,

                "label": label,

                "vit": vit_score,

                "ae": ae_score,

                "ecc_modules": ecc_mod,

                "ecc_syndromes": ecc_syn,

                "ecc_residual": ecc_res
            })

            print(
                f"{number:03d}    "
                f"{label:<12}"
                f"{vit_score:<14.6f}"
                f"{ae_score:<14.6f}"
                f"{ecc_mod:<12}"
                f"{ecc_syn:<12}"
                f"{ecc_res:<15}"
            )

    print(
        "================================================================================================"
    )

    genuine = [
        r for r in results
        if r["label"] == "Genuine"
    ]

    tampered = [
        r for r in results
        if r["label"] == "Tampered"
    ]

    print()
    print(
        "SUMMARY"
    )
    print(
        "-------"
    )

    print(
        "Genuine samples:",
        len(genuine)
    )

    print(
        "Tampered samples:",
        len(tampered)
    )

    if genuine:

        print()

        print(
            "Genuine average ViT:",
            f"{np.mean([r['vit'] for r in genuine]):.6f}"
        )

        print(
            "Genuine average AE:",
            f"{np.mean([r['ae'] for r in genuine]):.6f}"
        )

        print(
            "Genuine average ECC:",
            f"{np.mean([r['ecc_residual'] for r in genuine]):.2f}"
        )

    if tampered:

        print()

        print(
            "Tampered average ViT:",
            f"{np.mean([r['vit'] for r in tampered]):.6f}"
        )

        print(
            "Tampered average AE:",
            f"{np.mean([r['ae'] for r in tampered]):.6f}"
        )

        print(
            "Tampered average ECC:",
            f"{np.mean([r['ecc_residual'] for r in tampered]):.2f}"
        )

    print()

    print(
        "VALIDATION BRANCH EVALUATION COMPLETE."
    )


if __name__ == "__main__":
    main()
