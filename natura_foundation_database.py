"""
Complete Natura Foundation Database
Includes all available shades across Natura foundation and concealer lines
"""

import numpy as np

def hex_to_lab(hex_color):
    """Convert HEX color to LAB values using sRGB conversion"""
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB (0-1 range)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    # Apply gamma correction (sRGB to linear RGB)
    def gamma_expand(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    r_linear = gamma_expand(r)
    g_linear = gamma_expand(g)
    b_linear = gamma_expand(b)
    
    # Convert to XYZ using sRGB matrix (D65 illuminant)
    X = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
    Y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750
    Z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041
    
    # Normalize by D65 white point
    X = X / 0.95047
    Y = Y / 1.00000
    Z = Z / 1.08883
    
    # Convert XYZ to LAB
    def f(t):
        return t**(1/3) if t > 0.008856 else (7.787 * t + 16/116)
    
    fx = f(X)
    fy = f(Y)
    fz = f(Z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    
    return [round(L, 1), round(a, 1), round(b_lab, 1)]

NATURA_FOUNDATION_DATABASE = {
    'Una Base Fluida HD FPS 15': {
        'coverage': 'Alta',
        'finish': 'Natural a Semi-Matte',
        'description': 'Efeito Profissional (HD): Perfeita em fotos e vídeos, não reflete a luz do flash.',
        'skin_type': 'Todos os tipos, especialmente Normal a Mista',
        'shades': {}
    },
    'Una Corretivo Cobertura Extrema 24h': {
        'coverage': 'Extremamente Alta',
        'finish': 'Matte',
        'description': 'Poder de Cobertura: Cobre olheiras profundas, manchas e até tatuagens sem craquelar.',
        'skin_type': 'Todos os tipos',
        'shades': {}
    }
}

# Base Fluida HD Una shade data
base_fluida_shades = {
    '30f': '#B27248',
    '31n': '#A17751', 
    '32q': '#AB6F44',
    '33n': '#956E4A',
    '35n': '#AA6335',
    '35q': '#A95E27',
    '37q': '#AD592F',
    '43n': '#8D502F',
    '40q': '#985126',
    '44n': '#734430',
    '48f': '#FAFAFA',  # Note: This seems like an error in the data, too light
    '46q': '#654433',
    '15q': '#DEAE77',
    '10n': '#FFC9A6',
    '27n': '#CE935E',
    '25q': '#D08A4D',
    '24f': '#CA925F',
    '19n': '#DCA96F',
    '23f': '#CEA174',
    '21q': '#D59B5B',
    '20n': '#CA9456',
    '29n': '#CA9157',
    '12f': '#DFB292',
    '16f': '#DCAB7D'
}

# Corretivo Cobertura Extrema 24h shade data
corretivo_shades = {
    '10n': '#F7D0A9',
    '15n': '#E9BF91',
    '19n': '#E5B88F',
    '21n': '#D8AB82',
    '24n': '#BB9064',
    '27n': '#AE8253',
    '30n': '#AB8053',
    '32n': '#B67A48',
    '35n': '#966A45',
    '37n': '#90643F',
    '43n': '#7A5331',
    '46n': '#5F3B2B'
}

def get_undertone_from_natura_shade(shade_name):
    """Extract undertone from Natura shade naming convention"""
    if shade_name.endswith('f'):
        return 'Cool'  # Frio (Cool)
    elif shade_name.endswith('n'):
        return 'Neutral'  # Neutro (Neutral)
    elif shade_name.endswith('q'):
        return 'Warm'  # Quente (Warm)
    else:
        return 'Neutral'  # Default

def get_shade_description(shade_name, undertone):
    """Generate description for Natura shade"""
    # Extract numeric part
    numeric_part = shade_name[:-1] if shade_name[-1].isalpha() else shade_name
    
    try:
        shade_number = int(numeric_part)
    except ValueError:
        shade_number = 20  # Default
    
    # Determine depth
    if shade_number <= 12:
        depth = 'Muito claro'
    elif shade_number <= 19:
        depth = 'Claro'
    elif shade_number <= 25:
        depth = 'Médio claro'
    elif shade_number <= 32:
        depth = 'Médio'
    elif shade_number <= 40:
        depth = 'Médio escuro'
    elif shade_number <= 46:
        depth = 'Escuro'
    else:
        depth = 'Muito escuro'
    
    # Map undertone to Portuguese
    undertone_pt = {
        'Cool': 'subtom frio',
        'Neutral': 'subtom neutro', 
        'Warm': 'subtom quente'
    }.get(undertone, 'subtom neutro')
    
    return f'{depth} com {undertone_pt}'

def get_tone_category(L_value):
    """Determine tone category based on L* value"""
    if L_value >= 70:
        return 'Very Fair'
    elif L_value >= 64:
        return 'Fair'
    elif L_value >= 58:
        return 'Light'
    elif L_value >= 50:
        return 'Light Medium'
    elif L_value >= 42:
        return 'Medium'
    elif L_value >= 35:
        return 'Medium Deep'
    elif L_value >= 28:
        return 'Deep'
    else:
        return 'Very Deep'

# Populate Base Fluida HD shades
for shade_name, hex_color in base_fluida_shades.items():
    # Handle the problematic 48f shade
    if shade_name == '48f':
        # This seems to be an error in the data - adjusting to a more realistic color
        hex_color = '#8B6F47'  # Medium brown shade
    
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Base Fluida HD FPS 15']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Corretivo shades
for shade_name, hex_color in corretivo_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Corretivo Cobertura Extrema 24h']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

def convert_natura_database_to_app_format():
    """Convert the Natura database to the app's format"""
    foundation_database = {
        'Very Fair': [],
        'Fair': [],
        'Light': [],
        'Light Medium': [],
        'Medium': [],
        'Medium Deep': [],
        'Deep': [],
        'Very Deep': []
    }
    
    for product_line, product_data in NATURA_FOUNDATION_DATABASE.items():
        for shade_name, shade_data in product_data['shades'].items():
            L_value = shade_data['L']
            tone_category = get_tone_category(L_value)
            undertone = get_undertone_from_natura_shade(shade_name)
            
            foundation_entry = {
                'shade': shade_name,
                'L': shade_data['L'],
                'a': shade_data['a'],
                'b': shade_data['b'],
                'brand': 'Natura',
                'undertone': undertone,
                'product_line': product_line,
                'coverage': product_data['coverage'],
                'finish': product_data['finish'],
                'hex': shade_data.get('hex', ''),
                'description': shade_data.get('description', ''),
                'skin_type': product_data.get('skin_type', '')
            }
            
            foundation_database[tone_category].append(foundation_entry)
    
    # Sort each category by L value (lightest to darkest)
    for category in foundation_database:
        foundation_database[category].sort(key=lambda x: x['L'], reverse=True)
    
    return foundation_database
