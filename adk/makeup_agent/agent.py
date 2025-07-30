import logging
import base64
import requests
import json
from io import BytesIO
import google.genai.types as types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# --- Callback to save uploaded image as artifact ---
async def _save_uploaded_image_as_artifact(callback_context: CallbackContext):
    """Extracts image data from incoming message and saves as artifact."""
    logger.info("--- Entering _save_uploaded_image_as_artifact callback ---")
    user_content = callback_context.user_content

    if not user_content or not user_content.parts:
        logger.info("Callback: No content or parts found in user_content.")
        return

    # Look for image in user content
    for i, part in enumerate(user_content.parts):
        if hasattr(part, 'inline_data') and getattr(part.inline_data, 'mime_type', '').startswith('image/'):
            mime_type = part.inline_data.mime_type
            logger.info(f"Found image with mime_type: {mime_type}")
            
            # Extract image data
            image_bytes = None
            if hasattr(part.inline_data, 'data') and isinstance(part.inline_data.data, (bytes, bytearray)):
                image_bytes = part.inline_data.data
            elif hasattr(part.inline_data, 'data') and isinstance(part.inline_data.data, str):
                try:
                    # Try to decode if it's base64
                    image_bytes = base64.b64decode(part.inline_data.data)
                except:
                    logger.error("Failed to decode image data")
                    continue
            
            if image_bytes:
                try:
                    # Create artifact from image
                    image_artifact = types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    )
                    
                    # Save as artifact with a fixed filename
                    filename = "user_photo.jpg"
                    version = await callback_context.save_artifact(
                        filename=filename,
                        artifact=image_artifact
                    )
                    logger.info(f"Successfully saved image as artifact '{filename}' version {version}")
                    break
                    
                except ValueError as e:
                    logger.error(f"Error saving artifact: {e}. Is ArtifactService configured in Runner?")
                except Exception as e:
                    logger.error(f"Unexpected error saving artifact: {e}")
    
    logger.info("--- Exiting _save_uploaded_image_as_artifact callback ---")

# --- Tool to analyze skin tone via API ---
async def analyze_skin_tone_tool(tool_context: ToolContext) -> str:
    """
    Analisa o tom de pele usando a imagem salva como artifact.
    
    Args:
        tool_context: The tool context with access to artifacts
        
    Returns:
        str: Análise formatada em português com recomendações
    """
    logger.info("Starting skin tone analysis")
    
    try:
        # Load the user photo artifact
        filename = "user_photo.jpg"
        photo_artifact = await tool_context.load_artifact(filename=filename)
        
        if not photo_artifact or not photo_artifact.inline_data:
            logger.warning(f"No photo artifact found with filename '{filename}'")
            return "Não encontrei nenhuma imagem para analisar. Por favor, envie uma foto do seu rosto para que eu possa ajudar você a escolher o tom ideal de maquiagem."
        
        logger.info(f"Successfully loaded photo artifact. MIME Type: {photo_artifact.inline_data.mime_type}")
        
        # Get image bytes from artifact
        image_bytes = photo_artifact.inline_data.data
        logger.info(f"Photo size: {len(image_bytes)} bytes")
        
        # Prepare the request to the API
        files = {
            'image': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')
        }
        data = {
            'num_matches': '10',
            'include_statistics': 'true'
        }
        
        # Make the API request
        logger.info("Calling skin analysis API...")
        response = requests.post(
            'http://localhost:9090/api/v1/analyze',
            files=files,
            data=data,
            timeout=30
        )
        
        # Check response status
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}: {response.text}")
            
            # Parse error response if possible
            try:
                error_data = response.json()
                if error_data.get('error', {}).get('code') == 'NO_SKIN_DETECTED':
                    return (
                        "😕 Não consegui detectar sua pele na imagem.\n\n"
                        "Algumas dicas para uma melhor análise:\n"
                        "• Certifique-se de que seu rosto está bem iluminado\n"
                        "• Evite sombras no rosto\n"
                        "• Tire a foto de frente, com o rosto claramente visível\n"
                        "• Se possível, remova os óculos\n\n"
                        "Por favor, tente enviar outra foto!"
                    )
            except Exception:
                pass
                
            return "Desculpe, não consegui analisar a imagem. Por favor, tente novamente com outra foto."
        
        # Parse the successful response
        result = response.json()
        
        if not result.get('success') or 'data' not in result:
            return "Houve um problema ao processar a análise. Por favor, tente novamente."
        
        data = result['data']
        skin_analysis = data.get('skin_analysis', {})
        matches = data.get('matches', [])
        
        # Format the response in Portuguese
        response_text = "✨ **Análise do seu tom de pele concluída!** ✨\n\n"
        
        # Skin color analysis
        color_info = skin_analysis.get('color', {})
        if color_info:
            hex_color = color_info.get('hex', '#000000')
            response_text += f"**Cor detectada:** {hex_color}\n"
        
        # Undertone analysis
        undertone_info = skin_analysis.get('undertone', {})
        if undertone_info:
            undertone = undertone_info.get('primary', 'Neutro')
            confidence = undertone_info.get('confidence', 0)
            
            undertone_pt = {
                'Warm': 'Quente',
                'Cool': 'Frio',
                'Neutral': 'Neutro'
            }.get(undertone, undertone)
            
            response_text += f"**Subtom:** {undertone_pt} (confiança: {confidence:.0f}%)\n\n"
        
        # Product recommendations
        if matches:
            response_text += "🎯 **Produtos recomendados especialmente para você:**\n\n"
            
            # Sort matches by rank if not already sorted
            sorted_matches = sorted(matches, key=lambda x: x.get('rank', 999))
            
            # Show top recommendations with quick decision helper
            if len(sorted_matches) > 0:
                # First match - best recommendation
                best_match = sorted_matches[0]
                response_text += "🥇 **MELHOR OPÇÃO PARA VOCÊ:**\n"
                response_text += format_product_recommendation(best_match, is_best=True)
                response_text += "\n"
            
            # Show alternatives if available
            if len(sorted_matches) > 1:
                response_text += "📋 **Outras opções que combinam com você:**\n\n"
                
                # Group by product line for easier comparison
                product_groups = {}
                for match in sorted_matches[1:10]:  # Show up to 9 more (total 10)
                    product_line = match.get('product_line', 'Produto')
                    if product_line not in product_groups:
                        product_groups[product_line] = []
                    product_groups[product_line].append(match)
                
                # Display grouped products
                for product_line, group_matches in product_groups.items():
                    response_text += f"**{product_line}**\n"
                    for match in group_matches[:3]:  # Max 3 per product line
                        response_text += format_product_recommendation(match, compact=True)
                    response_text += "\n"
            
            # Quick decision guide
            response_text += "💡 **Dica para escolha rápida:**\n"
            if sorted_matches[0].get('match_percentage', 0) >= 80:
                response_text += "• A primeira opção tem uma compatibilidade excelente com seu tom! É a escolha mais segura.\n"
            elif sorted_matches[0].get('match_percentage', 0) >= 70:
                response_text += "• A primeira opção tem boa compatibilidade. Se preferir um tom mais claro ou escuro, veja as alternativas.\n"
            else:
                response_text += "• Recomendo testar o produto na loja para confirmar o tom ideal, pois a iluminação da foto pode afetar o resultado.\n"
            
            # Add product-specific tips
            response_text += "\n💄 **Dicas para os produtos sugeridos:**\n"
            if any('Mate' in m.get('product_line', '') for m in sorted_matches[:3]):
                response_text += "• Base Mate: Ideal para pele oleosa, com acabamento sequinho\n"
            if any('Glycolic' in m.get('product_line', '') for m in sorted_matches[:3]):
                response_text += "• Base Glycolic com FPS 50: Protege do sol e trata a pele\n"
            if any('Intense' in m.get('product_line', '') for m in sorted_matches[:3]):
                response_text += "• Intense Camuflagem: Alta cobertura para ocasiões especiais\n"
            
            # Personalized tip based on undertone
            response_text += f"\n✨ **Dica especial para seu subtom {undertone_pt.lower()}:**\n"
            if undertone_pt == 'Quente':
                response_text += "• Valorize seu subtom com blushes pêssego, batons coral e iluminador dourado\n"
            elif undertone_pt == 'Frio':
                response_text += "• Realce sua beleza com blushes rosados, batons berry e iluminador prateado\n"
            else:
                response_text += "• Você tem liberdade total! Tanto tons quentes quanto frios ficam lindos em você\n"
        else:
            response_text += "Não foram encontradas correspondências. Por favor, tente com outra foto em melhor iluminação."
        
        return response_text
        
    except ValueError as e:
        logger.error(f"Error loading artifact: {e}. Is ArtifactService configured?")
        return "Erro ao carregar a imagem. Certifique-se de que o serviço está configurado corretamente."
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return "A análise está demorando mais que o esperado. Por favor, tente novamente."
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API")
        return "Não consegui conectar ao serviço de análise. Verifique se o servidor está rodando em http://localhost:9090"
    except Exception as e:
        logger.error(f"Unexpected error in analyze_skin_tone_tool: {e}", exc_info=True)
        return f"Ocorreu um erro inesperado durante a análise. Por favor, tente novamente."

def format_product_recommendation(match: dict, is_best: bool = False, compact: bool = False) -> str:
    """Format a single product recommendation in Portuguese"""
    product_line = match.get('product_line', 'Produto')
    shade = match.get('shade', '')
    shade_name = match.get('shade_name', '')
    match_percentage = match.get('match_percentage', 0)
    rank = match.get('rank', 0)
    
    if is_best:
        # Detailed format for the best match
        text = f"📦 **{product_line}**\n"
        text += f"🎨 Tom {shade} - {shade_name}\n"
        text += f"✅ Compatibilidade: {match_percentage:.0f}%"
        
        # Add quality indicator
        if match_percentage >= 90:
            text += " (Perfeito! 🤩)"
        elif match_percentage >= 80:
            text += " (Excelente! 😍)"
        elif match_percentage >= 70:
            text += " (Muito bom! 😊)"
        else:
            text += " (Bom 👍)"
        text += "\n"
        
    elif compact:
        # Compact format for alternatives
        text = f"   • Tom {shade}: {shade_name} ({match_percentage:.0f}%)\n"
    else:
        # Standard format
        text = f"• **{product_line}**\n"
        text += f"  Tom {shade}: {shade_name}\n"
        text += f"  Compatibilidade: {match_percentage:.0f}%\n\n"
    
    return text

# Create the Makeup Assistant Agent
makeup_assistant = Agent(
    name="assistente_maquiagem_boticario",
    model="gemini-2.5-flash",
    description="Assistente especializado em maquiagem da Boticário que analisa tom de pele e recomenda produtos",
    instruction=(
        "Você é a assistente virtual de maquiagem da Boticário! "
        "Seu papel é ajudar clientes a encontrar o tom perfeito de base e outros produtos de maquiagem. "
        "Você é amigável, profissional e conhece profundamente os produtos da marca.\n\n"
        
        "Quando o usuário enviar uma imagem:\n"
        "1. A imagem será automaticamente salva como artifact pelo callback\n"
        "2. Use a ferramenta analyze_skin_tone_tool para analisar o tom de pele\n"
        "3. Apresente os resultados de forma clara e amigável\n"
        "4. Explique as recomendações e dê dicas personalizadas\n\n"
        
        "Se o usuário não enviar imagem:\n"
        "- Peça educadamente para enviar uma foto do rosto\n"
        "- Explique que precisa da foto para fazer a análise correta\n"
        "- Dê dicas sobre como tirar uma boa foto (boa iluminação, rosto visível)\n\n"
        
        "Sempre:\n"
        "- Use emojis para tornar a conversa mais leve 💄\n"
        "- Seja entusiasta sobre maquiagem\n"
        "- Foque nos produtos da Boticário\n"
        "- Responda sempre em português brasileiro\n"
        "- Se houver erro na análise, sugira tentar novamente com outra foto"
        "- Responda sempre com uma lista ordenada dos produtos mais compatíveis e coloque sempre os detalhes do produto. "
    ),
    tools=[analyze_skin_tone_tool],
    before_agent_callback=_save_uploaded_image_as_artifact
)

logger.info(f"Makeup assistant agent '{makeup_assistant.name}' initialized with analyze_skin_tone_tool and artifact-based image handling.")

# Export the agent
root_agent = makeup_assistant
