import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mac_foundation_database import convert_mac_database_to_app_format
from app import create_color_swatch
import base64
from PIL import Image, ImageDraw, ImageFont
import io

def create_comparison_image():
    """Create a comparison image showing different foundation swatches side by side"""
    
    # Get foundation database
    foundation_db = convert_mac_database_to_app_format()
    
    # Find specific shades
    test_shades = ['NC15', 'NC20', 'NC25', 'NC30', 'NC35', 'NC40']
    found_foundations = []
    
    for category, foundations in foundation_db.items():
        for foundation in foundations:
            if foundation['shade'] in test_shades:
                found_foundations.append(foundation)
    
    # Sort by shade name
    found_foundations.sort(key=lambda x: x['shade'])
    
    # Create comparison image
    swatch_width = 150
    swatch_height = 100
    padding = 20
    
    img_width = len(found_foundations) * (swatch_width + padding) + padding
    img_height = swatch_height + 80  # Extra space for text
    
    comparison_img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(comparison_img)
    
    # Draw each swatch
    x_offset = padding
    for foundation in found_foundations:
        L, a, b = foundation['L'], foundation['a'], foundation['b']
        
        # Convert LAB to RGB manually for direct color
        # LAB to XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        def f_inv(t):
            delta = 6/29
            return t**3 if t > delta else 3 * delta**2 * (t - 4/29)
        
        X = 95.047 * f_inv(fx) / 100.0
        Y = 100.0 * f_inv(fy) / 100.0
        Z = 108.883 * f_inv(fz) / 100.0
        
        # XYZ to sRGB
        r = X *  3.2406 + Y * -1.5372 + Z * -0.4986
        g = X * -0.9689 + Y *  1.8758 + Z *  0.0415
        b_rgb = X *  0.0557 + Y * -0.2040 + Z *  1.0570
        
        # Gamma correction
        def gamma_correct(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c**(1/2.4)) - 0.055
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b_rgb = gamma_correct(b_rgb)
        
        # Final RGB
        rgb = tuple(int(max(0, min(255, c * 255))) for c in [r, g, b_rgb])
        
        # Draw swatch
        draw.rectangle(
            [x_offset, 20, x_offset + swatch_width, 20 + swatch_height],
            fill=rgb,
            outline='black',
            width=2
        )
        
        # Add text
        text = f"{foundation['shade']}\nRGB{rgb}\nLAB({L:.0f},{a:.0f},{b:.0f})"
        # Split text into lines and draw each separately
        lines = text.split('\n')
        y_text = swatch_height + 30
        for line in lines:
            # Get text size for centering
            bbox = draw.textbbox((0, 0), line)
            text_width = bbox[2] - bbox[0]
            draw.text((x_offset + (swatch_width - text_width)//2, y_text), line, fill='black')
            y_text += 15
        
        x_offset += swatch_width + padding
    
    # Save comparison image
    comparison_img.save('foundation_comparison.png')
    print("Created foundation_comparison.png showing different shades side by side")
    
    # Also check if the base64 swatches are unique
    print("\nChecking base64 uniqueness:")
    base64_swatches = {}
    for foundation in found_foundations[:4]:
        swatch_b64 = create_color_swatch([foundation['L'], foundation['a'], foundation['b']])
        base64_swatches[foundation['shade']] = swatch_b64[:50]  # First 50 chars
    
    for shade, b64_start in base64_swatches.items():
        print(f"{shade}: {b64_start}...")
        
    # Check if any are identical
    unique_swatches = len(set(base64_swatches.values()))
    total_swatches = len(base64_swatches)
    print(f"\nUnique swatches: {unique_swatches}/{total_swatches}")
    
    if unique_swatches < total_swatches:
        print("WARNING: Some swatches have identical base64 encodings!")

if __name__ == "__main__":
    create_comparison_image()
