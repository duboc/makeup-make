import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_color_swatch
from mac_foundation_database import convert_mac_database_to_app_format
import base64
from PIL import Image
import io
import numpy as np

def test_lab_conversion():
    """Test LAB conversion for specific foundation shades"""
    
    # Get foundation database
    foundation_db = convert_mac_database_to_app_format()
    
    # Find specific shades from the screenshot
    test_shades = ['NC20', 'NC25', 'NC15']
    found_foundations = []
    
    for category, foundations in foundation_db.items():
        for foundation in foundations:
            if foundation['shade'] in test_shades:
                found_foundations.append(foundation)
    
    print("Testing LAB to RGB conversion for foundation shades:")
    print("=" * 60)
    
    for foundation in found_foundations:
        shade = foundation['shade']
        L = foundation['L']
        a = foundation['a']
        b = foundation['b']
        
        print(f"\nShade: {shade}")
        print(f"LAB values: L={L:.1f}, a={a:.1f}, b={b:.1f}")
        
        # Create swatch using the app function
        swatch_b64 = create_color_swatch([L, a, b])
        
        # Decode and analyze the image
        try:
            img_data = base64.b64decode(swatch_b64)
            img = Image.open(io.BytesIO(img_data))
            
            # Get the color from the image
            pixels = np.array(img)
            # Get the first pixel (they should all be the same)
            rgb = pixels[0, 0]
            
            print(f"Resulting RGB: R={rgb[0]}, G={rgb[1]}, B={rgb[2]}")
            print(f"Hex color: #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            
        except Exception as e:
            print(f"Error analyzing swatch: {e}")
    
    # Test manual LAB to RGB conversion
    print("\n" + "=" * 60)
    print("Manual LAB to RGB conversion test:")
    
    for foundation in found_foundations[:3]:  # Test first 3
        L, a, b = foundation['L'], foundation['a'], foundation['b']
        print(f"\n{foundation['shade']}: LAB({L:.1f}, {a:.1f}, {b:.1f})")
        
        # Manual conversion
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
        
        print(f"XYZ: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
        
        # XYZ to sRGB
        r = X *  3.2406 + Y * -1.5372 + Z * -0.4986
        g = X * -0.9689 + Y *  1.8758 + Z *  0.0415
        b_rgb = X *  0.0557 + Y * -0.2040 + Z *  1.0570
        
        print(f"Linear RGB: r={r:.3f}, g={g:.3f}, b={b_rgb:.3f}")
        
        # Gamma correction
        def gamma_correct(c):
            return 12.92 * c if c <= 0.0031308 else 1.055 * (c**(1/2.4)) - 0.055
        
        r_gamma = gamma_correct(r)
        g_gamma = gamma_correct(g)
        b_gamma = gamma_correct(b_rgb)
        
        print(f"Gamma corrected: r={r_gamma:.3f}, g={g_gamma:.3f}, b={b_gamma:.3f}")
        
        # Final RGB
        rgb_final = np.clip([r_gamma, g_gamma, b_gamma], 0, 1) * 255
        rgb_final = rgb_final.astype(int)
        
        print(f"Final RGB: R={rgb_final[0]}, G={rgb_final[1]}, B={rgb_final[2]}")
        print(f"Hex: #{rgb_final[0]:02x}{rgb_final[1]:02x}{rgb_final[2]:02x}")

if __name__ == "__main__":
    test_lab_conversion()
