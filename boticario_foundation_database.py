"""
Complete O Boticário Foundation Database
Includes Quem Disse, Berenice? and Make B. foundation lines
Updated with actual hex color values for accurate matching
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

BOTICARIO_FOUNDATION_DATABASE = {
    # QDB (Quem Disse, Berenice?) Product Lines
    'QDB Base Líquida Tô No Glow 30ml': {
        'coverage': 'Média',
        'finish': 'Natural Radiante',
        'description': 'Base com glow natural, cobertura média e hidratação sem oleosidade',
        'skin_type': 'Todos os tipos',
        'shades': {}
    },
    
    # Make B. Specialized Lines
    'Make B. Base Líquida Mate Salicylic 30g': {
        'coverage': 'Média a Alta',
        'finish': 'Matte',
        'description': 'Base matificante com ácido salicílico que controla oleosidade',
        'skin_type': 'Pele Oleosa a Mista',
        'shades': {}
    },
    'Make B. Base Líquida Glycolic TX FPS 50 30g': {
        'coverage': 'Média',
        'finish': 'Natural',
        'description': 'Base anti-manchas com ácido glicólico e tranexâmico',
        'skin_type': 'Todos os tipos',
        'shades': {}
    },
    'Make B. Base Líquida Retinol H+ FPS 80 26g': {
        'coverage': 'Média a Alta',
        'finish': 'Natural Luminoso',
        'description': 'Base anti-idade com retinol e ácido hialurônico',
        'skin_type': 'Pele Madura',
        'shades': {}
    },
    'Make B. Base em Pó Mineral 5,5g': {
        'coverage': 'Leve a Média',
        'finish': 'Matte Mineral',
        'description': 'Base em pó mineral com cobertura natural',
        'skin_type': 'Pele Sensível e Oleosa',
        'shades': {}
    },
    
    # Intense Line (O Boticário)
    'Intense Base Mate Camuflagem Pop! 20ml': {
        'coverage': 'Alta',
        'finish': 'Matte',
        'description': 'Base de alta cobertura com efeito camuflagem',
        'skin_type': 'Todos os tipos',
        'shades': {}
    }
}

# QDB Tô No Glow shade data - ACTUAL hex values
qdb_to_no_glow_shades = {
    '330N': '#65372A',
    '310Q': '#955440',
    '300N': '#8C4E35',
    '292O': '#9E623D',
    '290F': '#A06440',
    '280Q': '#DA9866',
    '270Q': '#E1A370',
    '260O': '#C88D6B',
    '250Q': '#DBA06E',
    '240F': '#C18B65',
    '230Q': '#E0AB79',
    '220O': '#CB9772',
    '210N': '#E2AA77',
    '200F': '#DFB185',
    '190N': '#F1BE92',
    '180Q': '#E2AD83',
    '150F': '#EBBA92',
    '130Q': '#F0C9A5',
    '120N': '#E6C3A7',
    '100F': '#EDC8AA'
}

# Make B. Glycolic TX shade data - ACTUAL hex values
make_b_glycolic_tx_shades = {
    '320': '#572911',
    '270': '#A46A44',
    '330': '#3C1E13',
    '300': '#542B14',
    '260': '#9C6B4A',
    '230': '#A8744F',
    '220': '#996C4F',
    '210': '#AC7B57',
    '180': '#AF7E60',
    '190': '#AA7D5E',
    '140': '#BB977B',
    '130': '#BF9678',
    '120': '#BC967F',
    '100': '#BA9A85'
}

# Make B. Mate Salicylic shade data - ACTUAL hex values
make_b_mate_salicylic_shades = {
    '320': '#623425',
    '330': '#542A1C',
    '310': '#7D442C',
    '300': '#734838',
    '290': '#78462B',
    '280': '#995B36',
    '270': '#A86C47',
    '260': '#A37051',
    '250': '#A46E4C',
    '240': '#9B694E',
    '230': '#B27B5C',
    '220': '#A4765C',
    '210': '#B48362',
    '200': '#A47D5E',
    '190': '#AE7E60',
    '180': '#B38166',
    '170': '#C79A71',
    '160': '#B58E71',
    '150': '#B38168',
    '140': '#BD9478',
    '130': '#CAA285',
    '120': '#BE9883',
    '110': '#C39E81',
    '100': '#C5A38F'
}

# Make B. Retinol H+ shade data - ACTUAL hex values
make_b_retinol_h_shades = {
    '80': '#4B1A04',
    '75': '#782A04',
    '70': '#823F25',
    '65': '#995234',
    '60': '#AD5B36',
    '53': '#B8764C',
    '50': '#CC7F51',
    '40': '#B78061',
    '30': '#BF835E',
    '25': '#D39973',
    '20': '#E1AA84',
    '15': '#E2AF80',
    '10': '#E3BB92'
}

# Make B. Base em Pó Mineral shade data - ACTUAL hex values
make_b_mineral_shades = {
    '70': '#915034',
    '60': '#9B6949',
    '50': '#995E3E',
    '40': '#BD8761',
    '30': '#DDA479',
    '20': '#D8AF84',
    '10': '#EAC39E'
}

# Intense Base Mate Camuflagem Pop! shade data - ACTUAL hex values
intense_camuflagem_shades = {
    '330': '#482F28',
    '320': '#73432C',
    '300': '#7F4A29',
    '260': '#AF7B54',
    '240': '#AA7B5D',
    '230': '#BF895B',
    '210': '#C08F64',
    '190': '#C3936F',
    '160': '#C59F7A',
    '140': '#DAB38F',
    '120': '#D5B090',
    '100': '#E3C1A6'
}

def get_undertone_from_boticario_shade(shade_name):
    """Extract undertone from Boticário shade naming convention"""
    if shade_name.endswith('F'):
        return 'Cool'  # Frio (Cool)
    elif shade_name.endswith('N'):
        return 'Neutral'  # Neutro (Neutral)
    elif shade_name.endswith('Q'):
        return 'Warm'  # Quente (Warm)
    elif shade_name.endswith('O'):
        return 'Olive'  # Oliva (Olive)
    else:
        return 'Neutral'  # Default for numbered shades
        
def get_shade_description(shade_name, undertone):
    """Generate description for Boticário shade"""
    # Extract numeric part
    numeric_part = ''.join(filter(str.isdigit, shade_name))
    
    try:
        shade_number = int(numeric_part)
    except ValueError:
        shade_number = 200  # Default to medium
    
    # Determine depth based on shade number
    if shade_number <= 120:
        depth = 'Muito claro'
    elif shade_number <= 160:
        depth = 'Claro'
    elif shade_number <= 200:
        depth = 'Médio claro'
    elif shade_number <= 250:
        depth = 'Médio'
    elif shade_number <= 300:
        depth = 'Médio escuro'
    elif shade_number <= 350:
        depth = 'Escuro'
    else:
        depth = 'Muito escuro'
    
    # Map undertone to Portuguese
    undertone_pt = {
        'Cool': 'subtom frio',
        'Neutral': 'subtom neutro', 
        'Warm': 'subtom quente',
        'Olive': 'subtom oliva'
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

# Populate QDB Tô No Glow shades with ACTUAL values
for shade_name, hex_color in qdb_to_no_glow_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_boticario_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    BOTICARIO_FOUNDATION_DATABASE['QDB Base Líquida Tô No Glow 30ml']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Make B. Mate Salicylic shades with ACTUAL values
for shade_name, hex_color in make_b_mate_salicylic_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_boticario_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    BOTICARIO_FOUNDATION_DATABASE['Make B. Base Líquida Mate Salicylic 30g']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Make B. Glycolic TX shades with ACTUAL values
for shade_name, hex_color in make_b_glycolic_tx_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_boticario_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    BOTICARIO_FOUNDATION_DATABASE['Make B. Base Líquida Glycolic TX FPS 50 30g']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Make B. Retinol H+ shades with ACTUAL values
for shade_name, hex_color in make_b_retinol_h_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_boticario_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    BOTICARIO_FOUNDATION_DATABASE['Make B. Base Líquida Retinol H+ FPS 80 26g']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Make B. Base em Pó Mineral shades with ACTUAL values
for shade_name, hex_color in make_b_mineral_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_boticario_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    BOTICARIO_FOUNDATION_DATABASE['Make B. Base em Pó Mineral 5,5g']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Intense Base Mate Camuflagem Pop! shades with ACTUAL values
for shade_name, hex_color in intense_camuflagem_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_boticario_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    BOTICARIO_FOUNDATION_DATABASE['Intense Base Mate Camuflagem Pop! 20ml']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

def convert_boticario_database_to_app_format():
    """Convert the Boticário database to the app's format"""
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
    
    for product_line, product_data in BOTICARIO_FOUNDATION_DATABASE.items():
        for shade_name, shade_data in product_data['shades'].items():
            L_value = shade_data['L']
            tone_category = get_tone_category(L_value)
            undertone = get_undertone_from_boticario_shade(shade_name)
            
            # Determine brand based on product line
            if 'QDB' in product_line:
                brand = 'Quem Disse, Berenice?'
            elif 'Intense' in product_line:
                brand = 'O Boticário'
            else:
                brand = 'Make B.'
            
            foundation_entry = {
                'shade': shade_name,
                'L': shade_data['L'],
                'a': shade_data['a'],
                'b': shade_data['b'],
                'brand': brand,
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
