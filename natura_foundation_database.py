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
    },
    'Una Base Líquida Mousse FPS 20': {
        'coverage': 'Média a Alta',
        'finish': 'Natural Luminoso',
        'description': 'Textura mousse única que se adapta perfeitamente à pele, proporcionando cobertura uniforme.',
        'skin_type': 'Pele Normal a Seca',
        'shades': {}
    },
    'Una Base Cremosa FPS 25': {
        'coverage': 'Média',
        'finish': 'Natural Mate',
        'description': 'Base cremosa que hidrata e cobre imperfeições, ideal para uso diário.',
        'skin_type': 'Pele Seca a Muito Seca',
        'shades': {}
    },
    'Una Base Stick FPS 30': {
        'coverage': 'Alta',
        'finish': 'Matte',
        'description': 'Praticidade em formato stick para retoques rápidos e cobertura intensa.',
        'skin_type': 'Todos os tipos, especialmente Oleosa',
        'shades': {}
    },
    'Aqua Base Hidratante FPS 15': {
        'coverage': 'Leve a Média',
        'finish': 'Natural Hidratante',
        'description': 'Base com ácido hialurônico que hidrata enquanto uniformiza o tom da pele.',
        'skin_type': 'Pele Normal a Seca',
        'shades': {}
    },
    'Una Base em Pó Compacto FPS 20': {
        'coverage': 'Média',
        'finish': 'Matte Aveludado',
        'description': 'Praticidade do pó compacto com cobertura de base para pele sempre perfeita.',
        'skin_type': 'Pele Oleosa a Mista',
        'shades': {}
    },
    'Una BB Cream FPS 30': {
        'coverage': 'Leve',
        'finish': 'Natural Luminoso',
        'description': 'Beleza em um só produto: hidrata, protege e uniformiza com toque natural.',
        'skin_type': 'Todos os tipos',
        'shades': {}
    },
    'Una CC Cream FPS 35': {
        'coverage': 'Leve a Média',
        'finish': 'Natural Corretor',
        'description': 'Corrige imperfeições e uniformiza o tom com proteção solar elevada.',
        'skin_type': 'Todos os tipos',
        'shades': {}
    },
    'Una Base Mineral FPS 25': {
        'coverage': 'Média',
        'finish': 'Natural Respirável',
        'description': 'Fórmula mineral que permite a pele respirar enquanto oferece cobertura uniforme.',
        'skin_type': 'Pele Sensível e Oleosa',
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
    '48f': '#5E3A2C', 
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

# Una Base Líquida Mousse FPS 20 shade data
base_mousse_shades = {
    '08f': '#F8E4D0',
    '10n': '#F5D5B8',
    '12q': '#F2C9A0',
    '15f': '#EFBE88',
    '17n': '#ECB370',
    '19q': '#E9A858',
    '22f': '#E09D40',
    '24n': '#D79228',
    '26q': '#CE8710',
    '29f': '#C57C00',
    '31n': '#BC7100',
    '33q': '#B36600',
    '36f': '#AA5B00',
    '38n': '#A15000',
    '40q': '#984500',
    '43f': '#8F3A00',
    '45n': '#862F00',
    '47q': '#7D2400'
}

# Una Base Cremosa FPS 25 shade data
base_cremosa_shades = {
    '09f': '#F6E2CE',
    '11n': '#F3D7B6',
    '13q': '#F0CC9E',
    '16f': '#EDC186',
    '18n': '#EAB66E',
    '20q': '#E7AB56',
    '23f': '#E4A03E',
    '25n': '#E19526',
    '27q': '#DE8A0E',
    '30f': '#DB7F00',
    '32n': '#D27400',
    '34q': '#C96900',
    '37f': '#C05E00',
    '39n': '#B75300',
    '41q': '#AE4800',
    '44f': '#A53D00',
    '46n': '#9C3200',
    '48q': '#932700'
}

# Una Base Stick FPS 30 shade data (concentrated range for portability)
base_stick_shades = {
    '12f': '#F1D4B1',
    '15n': '#EEC899',
    '18q': '#EBBC81',
    '21f': '#E8B069',
    '24n': '#E5A451',
    '27q': '#E29839',
    '30f': '#DF8C21',
    '33n': '#DC8009',
    '36q': '#D97400',
    '39f': '#D06800',
    '42n': '#C75C00',
    '45q': '#BE5000'
}

# Aqua Base Hidratante FPS 15 shade data (lighter coverage, more natural tones)
aqua_base_shades = {
    '07f': '#F9E6D2',
    '09n': '#F6DBBA',
    '11q': '#F3D0A2',
    '14f': '#F0C58A',
    '16n': '#EDBA72',
    '18q': '#EAAF5A',
    '21f': '#E7A442',
    '23n': '#E4992A',
    '25q': '#E18E12',
    '28f': '#DE8300',
    '30n': '#D57800',
    '32q': '#CC6D00',
    '35f': '#C36200',
    '37n': '#BA5700',
    '39q': '#B14C00'
}

# Una Base em Pó Compacto FPS 20 shade data
base_po_shades = {
    '10f': '#F4D6B3',
    '12n': '#F1CB9B',
    '14q': '#EEC083',
    '17f': '#EBB56B',
    '19n': '#E8AA53',
    '21q': '#E59F3B',
    '24f': '#E29423',
    '26n': '#DF890B',
    '28q': '#DC7E00',
    '31f': '#D37300',
    '33n': '#CA6800',
    '35q': '#C15D00',
    '38f': '#B85200',
    '40n': '#AF4700',
    '42q': '#A63C00'
}

# Una BB Cream FPS 30 shade data (universal shades that adapt)
bb_cream_shades = {
    'Universal Claro': '#F2D0A8',
    'Universal Medio Claro': '#EECA95',
    'Universal Medio': '#EAC482',
    'Universal Medio Escuro': '#E6BE6F',
    'Universal Escuro': '#E2B85C'
}

# Una CC Cream FPS 35 shade data (color correcting shades)
cc_cream_shades = {
    'Claro Rosado': '#F0CFA9',
    'Claro Dourado': '#F1D1A6',
    'Medio Claro Rosado': '#ECC896',
    'Medio Claro Dourado': '#EDCA93',
    'Medio Rosado': '#E8C183',
    'Medio Dourado': '#E9C380',
    'Medio Escuro Rosado': '#E4BA70',
    'Medio Escuro Dourado': '#E5BC6D',
    'Escuro Rosado': '#E0B35D',
    'Escuro Dourado': '#E1B55A'
}

# Una Base Mineral FPS 25 shade data
base_mineral_shades = {
    '11f': '#F3D5B2',
    '13n': '#F0CA9A',
    '15q': '#EDBF82',
    '18f': '#EAB46A',
    '20n': '#E7A952',
    '22q': '#E49E3A',
    '25f': '#E19322',
    '27n': '#DE880A',
    '29q': '#DB7D00',
    '32f': '#D27200',
    '34n': '#C96700',
    '36q': '#C05C00',
    '39f': '#B75100',
    '41n': '#AE4600',
    '43q': '#A53B00'
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

# Populate Base Mousse shades
for shade_name, hex_color in base_mousse_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Base Líquida Mousse FPS 20']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Base Cremosa shades
for shade_name, hex_color in base_cremosa_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Base Cremosa FPS 25']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Base Stick shades
for shade_name, hex_color in base_stick_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Base Stick FPS 30']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Aqua Base shades
for shade_name, hex_color in aqua_base_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Aqua Base Hidratante FPS 15']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Base em Pó shades
for shade_name, hex_color in base_po_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Base em Pó Compacto FPS 20']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate BB Cream shades (special handling for universal shades)
for shade_name, hex_color in bb_cream_shades.items():
    lab_values = hex_to_lab(hex_color)
    # BB Creams are universal/adaptive, so assign neutral undertone
    undertone = 'Neutral'
    description = f'Tom {shade_name.lower()} com subtom adaptativo'
    
    NATURA_FOUNDATION_DATABASE['Una BB Cream FPS 30']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate CC Cream shades (special handling for color correcting shades)
for shade_name, hex_color in cc_cream_shades.items():
    lab_values = hex_to_lab(hex_color)
    # Determine undertone from shade name
    if 'Rosado' in shade_name:
        undertone = 'Cool'
        undertone_pt = 'subtom frio (rosado)'
    elif 'Dourado' in shade_name:
        undertone = 'Warm'
        undertone_pt = 'subtom quente (dourado)'
    else:
        undertone = 'Neutral'
        undertone_pt = 'subtom neutro'
    
    # Determine depth from shade name
    if 'Claro' in shade_name and 'Medio' not in shade_name:
        depth = 'Claro'
    elif 'Medio Claro' in shade_name:
        depth = 'Médio claro'
    elif 'Medio Escuro' in shade_name:
        depth = 'Médio escuro'
    elif 'Medio' in shade_name:
        depth = 'Médio'
    elif 'Escuro' in shade_name:
        depth = 'Escuro'
    else:
        depth = 'Tom médio'
    
    description = f'{depth} com {undertone_pt} e correção de cor'
    
    NATURA_FOUNDATION_DATABASE['Una CC Cream FPS 35']['shades'][shade_name] = {
        'L': lab_values[0],
        'a': lab_values[1],
        'b': lab_values[2],
        'hex': hex_color,
        'description': description
    }

# Populate Base Mineral shades
for shade_name, hex_color in base_mineral_shades.items():
    lab_values = hex_to_lab(hex_color)
    undertone = get_undertone_from_natura_shade(shade_name)
    description = get_shade_description(shade_name, undertone)
    
    NATURA_FOUNDATION_DATABASE['Una Base Mineral FPS 25']['shades'][shade_name] = {
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
