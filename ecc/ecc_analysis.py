import cv2
import os
import numpy as np
import qrcode


VERSION = 3
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_M

GENUINE_DIR = "data/test/genuine"
TAMPERED_DIR = "data/test/tampered"

SIZE = 37
MODULE_SIZE = 10
SYMBOL_SIZE = 29

TOTAL_CODEWORDS = 70
DATA_CODEWORDS = 44
ECC_CODEWORDS = 26

RS_BLOCKS = 1
BLOCK_TOTAL = 70
BLOCK_DATA = 44
BLOCK_ECC = 26


# ============================================================
# GF(256) — QR CODE REED-SOLOMON FIELD
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
# QR MASK FUNCTIONS
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

    raise ValueError("Invalid QR mask pattern")


# ============================================================
# REFERENCE QR GENERATION
# ============================================================

def generate_reference(number):

    payload_index = number - 1

    qr = qrcode.QRCode(
        version=VERSION,
        error_correction=ERROR_CORRECTION,
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
# IMAGE MODULE SAMPLING
# ============================================================

def sample_modules(image):

    image = cv2.resize(
        image,
        (SIZE * MODULE_SIZE, SIZE * MODULE_SIZE),
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
# REMOVE QUIET ZONE
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

    format_coordinates = [
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

    for r, c in format_coordinates:

        bits <<= 1

        if matrix[r, c]:
            bits |= 1

    return bits


def decode_mask_pattern(matrix):

    format_value = format_bits_from_matrix(
        matrix
    )

    format_data = format_value ^ 0x5412

    data = format_data >> 10

    error_correction_bits = (
        data >> 3
    )

    mask = data & 0x07

    return mask, error_correction_bits


# ============================================================
# FUNCTIONAL MODULE MAP
# ============================================================

def build_functional_map():

    functional = np.zeros(
        (SYMBOL_SIZE, SYMBOL_SIZE),
        dtype=bool
    )

    # Finder patterns and separators
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

    # Timing patterns
    for c in range(8, SYMBOL_SIZE - 8):
        functional[6, c] = True

    for r in range(8, SYMBOL_SIZE - 8):
        functional[r, 6] = True

    # Version 3 alignment pattern
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

    # Format information
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

    # Dark module
    functional[
        SYMBOL_SIZE - 8,
        8
    ] = True

    return functional


# ============================================================
# DATA MODULE COORDINATES
# ============================================================

def get_data_coordinates():

    functional = build_functional_map()

    coordinates = []

    upward = True

    col = SYMBOL_SIZE - 1

    while col > 0:

        if col == 6:
            col -= 1

        rows = (
            range(
                SYMBOL_SIZE - 1,
                -1,
                -1
            )
            if upward
            else range(SYMBOL_SIZE)
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
# EXTRACT CODEWORDS
# ============================================================

def extract_codewords(matrix):

    symbol = remove_quiet_zone(matrix)

    mask, _ = decode_mask_pattern(
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

    for i in range(TOTAL_CODEWORDS):

        value = 0

        for j in range(8):

            value <<= 1

            value |= bits[
                i * 8 + j
            ]

        codewords.append(value)

    return codewords, mask


# ============================================================
# REED-SOLOMON SYNDROME
# ============================================================

def calculate_syndromes(codewords):

    if len(codewords) != BLOCK_TOTAL:
        raise ValueError(
            f"Expected {BLOCK_TOTAL} codewords, "
            f"got {len(codewords)}"
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

    return syndromes


# ============================================================
# ECC ANALYSIS
# ============================================================

def ecc_analysis(
    image,
    reference
):

    image_modules = sample_modules(
        image
    )

    module_difference = int(
        np.sum(
            image_modules
            != reference
        )
    )

    codewords, mask = extract_codewords(
        image_modules
    )

    syndromes = calculate_syndromes(
        codewords
    )

    nonzero_syndromes = sum(
        value != 0
        for value in syndromes
    )

    residual_magnitude = sum(
        syndromes
    )

    return {
        "module_difference":
            module_difference,

        "mask":
            mask,

        "syndromes":
            syndromes,

        "nonzero_syndromes":
            nonzero_syndromes,

        "residual_magnitude":
            residual_magnitude
    }


# ============================================================
# FIND TEST PAIRS
# ============================================================

def get_test_numbers():

    numbers = []

    for filename in os.listdir(
        GENUINE_DIR
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
            len("genuine_"):-len(".png")
        ]

        try:
            number = int(number_text)
        except ValueError:
            continue

        tampered_filename = (
            f"tampered_{number:03d}.png"
        )

        tampered_path = os.path.join(
            TAMPERED_DIR,
            tampered_filename
        )

        if os.path.exists(
            tampered_path
        ):
            numbers.append(number)

    return sorted(numbers)


# ============================================================
# MAIN
# ============================================================

def main():

    test_numbers = get_test_numbers()

    if len(test_numbers) == 0:
        raise ValueError(
            "No genuine/tampered test pairs found."
        )

    print()
    print(
        "FULL QR REED-SOLOMON ECC EVALUATION"
    )
    print(
        "==================================="
    )

    print()
    print("QR Configuration")
    print("-----------------")
    print("Version:", VERSION)
    print("Error Correction: M")
    print("Total codewords:", TOTAL_CODEWORDS)
    print("Data codewords:", DATA_CODEWORDS)
    print("ECC codewords:", ECC_CODEWORDS)
    print("RS blocks:", RS_BLOCKS)
    print("Codewords per block:", BLOCK_TOTAL)
    print("Data per block:", BLOCK_DATA)
    print("ECC per block:", BLOCK_ECC)

    print()
    print(
        "Test pairs found:",
        len(test_numbers)
    )

    print()
    print(
        "ECC TEST RESULTS"
    )
    print(
        "======================================================================"
    )

    print(
        f"{'Pair':<12}"
        f"{'GenuineMod':<14}"
        f"{'TamperedMod':<15}"
        f"{'GenuineSyn':<14}"
        f"{'TamperedSyn':<15}"
        f"{'GenuineRes':<14}"
        f"{'TamperedRes':<14}"
    )

    print(
        "----------------------------------------------------------------------"
    )

    genuine_pass = 0
    genuine_fail = 0

    tampered_detected = 0
    tampered_missed = 0

    results = []

    for number in test_numbers:

        genuine_filename = (
            f"genuine_{number:03d}.png"
        )

        tampered_filename = (
            f"tampered_{number:03d}.png"
        )

        genuine_path = os.path.join(
            GENUINE_DIR,
            genuine_filename
        )

        tampered_path = os.path.join(
            TAMPERED_DIR,
            tampered_filename
        )

        genuine = cv2.imread(
            genuine_path,
            cv2.IMREAD_GRAYSCALE
        )

        tampered = cv2.imread(
            tampered_path,
            cv2.IMREAD_GRAYSCALE
        )

        if genuine is None:
            print(
                f"ERROR: Could not read "
                f"{genuine_filename}"
            )
            continue

        if tampered is None:
            print(
                f"ERROR: Could not read "
                f"{tampered_filename}"
            )
            continue

        reference = generate_reference(
            number
        )

        try:

            genuine_result = ecc_analysis(
                genuine,
                reference
            )

            tampered_result = ecc_analysis(
                tampered,
                reference
            )

        except Exception as error:

            print()
            print(
                f"ERROR processing pair "
                f"{number:03d}: {error}"
            )

            continue

        genuine_syn = (
            genuine_result[
                "nonzero_syndromes"
            ]
        )

        tampered_syn = (
            tampered_result[
                "nonzero_syndromes"
            ]
        )

        genuine_res = (
            genuine_result[
                "residual_magnitude"
            ]
        )

        tampered_res = (
            tampered_result[
                "residual_magnitude"
            ]
        )

        if genuine_syn == 0:
            genuine_pass += 1
        else:
            genuine_fail += 1

        if tampered_syn > 0:
            tampered_detected += 1
        else:
            tampered_missed += 1

        results.append({
            "number": number,
            "genuine_modules":
                genuine_result[
                    "module_difference"
                ],
            "tampered_modules":
                tampered_result[
                    "module_difference"
                ],
            "genuine_syndromes":
                genuine_syn,
            "tampered_syndromes":
                tampered_syn,
            "genuine_residual":
                genuine_res,
            "tampered_residual":
                tampered_res
        })

        print(
            f"{number:03d}"
            f"{genuine_result['module_difference']:<14}"
            f"{tampered_result['module_difference']:<15}"
            f"{genuine_syn:<14}"
            f"{tampered_syn:<15}"
            f"{genuine_res:<14}"
            f"{tampered_res:<14}"
        )

    print(
        "======================================================================"
    )

    total = len(results)

    print()
    print(
        "SUMMARY"
    )
    print(
        "-------"
    )

    print(
        "Total test pairs:",
        total
    )

    print()
    print(
        "Genuine QR results"
    )

    print(
        "  PASS:",
        genuine_pass
    )

    print(
        "  FAIL:",
        genuine_fail
    )

    if total > 0:

        genuine_accuracy = (
            genuine_pass / total
        ) * 100

        print(
            f"  Validation rate: "
            f"{genuine_accuracy:.2f}%"
        )

    print()
    print(
        "Tampered QR results"
    )

    print(
        "  DETECTED:",
        tampered_detected
    )

    print(
        "  MISSED:",
        tampered_missed
    )

    if total > 0:

        tampered_detection_rate = (
            tampered_detected / total
        ) * 100

        print(
            f"  Detection rate: "
            f"{tampered_detection_rate:.2f}%"
        )

    print()

    if total > 0:

        overall_ecc_accuracy = (
            (
                genuine_pass
                + tampered_detected
            )
            / (2 * total)
        ) * 100

        print(
            f"ECC branch overall accuracy: "
            f"{overall_ecc_accuracy:.2f}%"
        )

    print()
    print(
        "FULL ECC EVALUATION COMPLETE."
    )


if __name__ == "__main__":
    main()
