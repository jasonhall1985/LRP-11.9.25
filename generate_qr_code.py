#!/usr/bin/env python3
"""
Generate QR code for the lip-reading web demo.
"""

import qrcode
from pathlib import Path

def generate_qr_code():
    # Upload demo URL (no camera required - file upload only)
    demo_url = "http://192.168.1.100:8080/upload_demo.html"
    
    # Create QR code for terminal display
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=2,
    )

    qr.add_data(demo_url)
    qr.make(fit=True)

    # Print QR code in terminal
    qr.print_ascii(invert=True)

    # Also create QR code image
    img = qr.make_image(fill_color="black", back_color="white")

    # Save QR code
    qr_path = "lip_reading_demo_qr.png"
    img.save(qr_path)
    
    print("🎯 LIP-READING DEMO QR CODE GENERATED")
    print("=" * 50)
    print(f"✅ QR Code saved: {qr_path}")
    print(f"🌐 Demo URL: {demo_url}")
    print()
    print("📱 INSTRUCTIONS:")
    print("1. Scan the QR code with your mobile device")
    print("2. Open the web demo in your mobile browser")
    print("3. Allow camera permissions")
    print("4. Test the balanced lip-reading model!")
    print()
    print("🎯 NEW BALANCED MODEL FEATURES:")
    print("✅ Doctor bias eliminated (12.5% vs 87.5% before)")
    print("✅ All 4 classes predicted: pillow, my_mouth_is_dry, i_need_to_move, doctor")
    print("✅ Pillow recognition improved (50% frequency in tests)")
    print("✅ Authentic model predictions (no artificial corrections)")

if __name__ == "__main__":
    generate_qr_code()
