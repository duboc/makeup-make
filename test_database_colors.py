"""Test script to verify the updated Boticário foundation database colors"""

from boticario_foundation_database import (
    BOTICARIO_FOUNDATION_DATABASE, 
    convert_boticario_database_to_app_format,
    hex_to_lab
)
import json

def test_color_conversion():
    """Test that hex colors are being converted to LAB correctly"""
    print("Testing color conversion for updated database...\n")
    
    # Test a few specific shades with known hex values
    test_shades = {
        'QDB Base Líquida Tô No Glow 30ml': {
            '100F': '#EDC8AA',
            '210N': '#E2AA77',
            '330N': '#65372A'
        },
        'Make B. Base Líquida Retinol H+ FPS 80 26g': {
            '10': '#E3BB92',
            '60': '#AD5B36',
            '80': '#4B1A04'
        }
    }
    
    print("Sample LAB conversions:")
    print("-" * 60)
    
    for product, shades in test_shades.items():
        print(f"\n{product}:")
        for shade_name, expected_hex in shades.items():
            # Get the shade from database
            if product in BOTICARIO_FOUNDATION_DATABASE:
                product_shades = BOTICARIO_FOUNDATION_DATABASE[product]['shades']
                if shade_name in product_shades:
                    shade_data = product_shades[shade_name]
                    print(f"  {shade_name}: HEX={shade_data['hex']} -> LAB=({shade_data['L']:.1f}, {shade_data['a']:.1f}, {shade_data['b']:.1f})")
                else:
                    print(f"  {shade_name}: NOT FOUND in database")

def analyze_database_distribution():
    """Analyze the distribution of colors in the database"""
    print("\n\nAnalyzing color distribution in database:")
    print("-" * 60)
    
    converted_db = convert_boticario_database_to_app_format()
    
    total_shades = 0
    for category, shades in converted_db.items():
        count = len(shades)
        total_shades += count
        if count > 0:
            print(f"\n{category}: {count} shades")
            # Show L value range
            l_values = [s['L'] for s in shades]
            print(f"  L range: {min(l_values):.1f} - {max(l_values):.1f}")
            
            # Show first and last shade
            if len(shades) > 0:
                first = shades[0]
                print(f"  Lightest: {first['brand']} {first['shade']} (L={first['L']:.1f})")
                last = shades[-1]
                print(f"  Darkest: {last['brand']} {last['shade']} (L={last['L']:.1f})")
    
    print(f"\nTotal shades in database: {total_shades}")

def check_specific_products():
    """Check that all products have the expected number of shades"""
    print("\n\nChecking product shade counts:")
    print("-" * 60)
    
    expected_counts = {
        'QDB Base Líquida Tô No Glow 30ml': 20,
        'Make B. Base Líquida Glycolic TX FPS 50 30g': 14,
        'Make B. Base Líquida Mate Salicylic 30g': 24,
        'Make B. Base Líquida Retinol H+ FPS 80 26g': 13,
        'Make B. Base em Pó Mineral 5,5g': 7,
        'Intense Base Mate Camuflagem Pop! 20ml': 12
    }
    
    for product_name, expected in expected_counts.items():
        if product_name in BOTICARIO_FOUNDATION_DATABASE:
            actual = len(BOTICARIO_FOUNDATION_DATABASE[product_name]['shades'])
            status = "✓" if actual == expected else "✗"
            print(f"{status} {product_name}: {actual} shades (expected {expected})")
        else:
            print(f"✗ {product_name}: NOT FOUND in database")

def verify_hex_colors():
    """Verify that all hex colors are valid"""
    print("\n\nVerifying hex color validity:")
    print("-" * 60)
    
    invalid_count = 0
    for product_name, product_data in BOTICARIO_FOUNDATION_DATABASE.items():
        for shade_name, shade_data in product_data['shades'].items():
            hex_color = shade_data.get('hex', '')
            if not hex_color or len(hex_color) != 7 or not hex_color.startswith('#'):
                print(f"Invalid hex color in {product_name} - {shade_name}: '{hex_color}'")
                invalid_count += 1
    
    if invalid_count == 0:
        print("✓ All hex colors are valid!")
    else:
        print(f"✗ Found {invalid_count} invalid hex colors")

if __name__ == "__main__":
    test_color_conversion()
    analyze_database_distribution()
    check_specific_products()
    verify_hex_colors()
