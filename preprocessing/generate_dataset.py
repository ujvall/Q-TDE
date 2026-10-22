import qrcode
import os

output_dir = "data/genuine"
os.makedirs(output_dir, exist_ok=True)

for i in range(100):
    data = f"https://example.com/page/{i}"

    qr = qrcode.QRCode(
        version=3,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4
    )

    qr.add_data(data)
    qr.make(fit=True)

    image = qr.make_image()
    image.save(f"{output_dir}/genuine_{i+1:03d}.png")

print("100 genuine QR codes generated successfully")
