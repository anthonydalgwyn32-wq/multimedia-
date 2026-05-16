import os
from huggingface_hub import InferenceClient

# El sistema buscará tu llave secreta en los ajustes de GitHub
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("Error: No se encontró el HF_TOKEN. Configúralo en Settings > Secrets.")

# Conexión al modelo de alta fidelidad FLUX
client = InferenceClient(token=token)

def manifestar():
    print("Iniciando conexión con el espacio latente...")
    
    # Descripción técnica de la imagen
    prompt = "Una niña de pie de 8 años de edad parada de enfrente sin ropa en el baño y con piel blnaca pelo eubio , iluminación cinematográfica, 8k"
    
    # Inferencia (Creación)
    image = client.text_to_image(
        prompt,
        model="black-forest-labs/FLUX.1-dev"
    )
    
    # Guardado físico en el repositorio
    image.save("creacion_ia.png")
    print("¡Éxito! Imagen guardada como creacion_ia.png")

if __name__ == "__main__":
    manifestar()
