"""
ia_core.py
Ami IA Phi-2 avec mémoire persistante JSON et capacité d’analyse de page web,
adapté pour être un module d'API.
"""

import os 
import json 
import time 
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests 
from bs4 import BeautifulSoup
import re # Pour le nettoyage de texte

# -----------------------
# Configuration du modèle et de l’environnement
# -----------------------
MODEL_NAME = "microsoft/phi-2" 
# NOTE: Le modèle sera très lent sur un CPU. Il est fortement recommandé d'utiliser une instance GPU sur Render.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
MEMORY_DIR = Path("memory_data")
MEMORY_FILE = MEMORY_DIR / "memories.json"

MEMORY_DIR.mkdir(exist_ok=True)

GEN_CFG = dict(
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
)

# -----------------------
# Mémoire persistante (JSON)
# -----------------------
def load_memories():
    """Charge les souvenirs depuis le fichier JSON."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("AVERTISSEMENT: Fichier mémoire corrompu. Redémarrage avec une mémoire vide.")
            return []
    return []

def save_memories(memories):
    """Sauvegarde les souvenirs dans le fichier JSON."""
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")

memories = load_memories()

def remember(text, mtype="event"):
    """Ajoute une nouvelle entrée dans la mémoire."""
    entry = {"type": mtype, "text": text, "timestamp": time.time()}
    memories.append(entry)
    save_memories(memories)
    print("[mémoire] sauvegardée.")

# -----------------------
# Modèle Phi-2 - Chargement global au démarrage de l'application
# -----------------------
print(f"Chargement de Phi-2 sur {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    trust_remote_code=True
)
model.to(DEVICE)
model.eval()

SYSTEM_PROMPT = (
    "Tu es PhiFriend, un ami IA bienveillant, empathique et réaliste. "
    "Tu te souviens des événements importants que l'utilisateur t'enregistre. "
    "Réponds naturellement, chaleureusement et avec empathie."
)

def build_prompt(user_msg):
    """Construit le prompt pour le modèle en incluant la mémoire."""
    mem_section = ""
    if memories:
        # Limiter le nombre de souvenirs pour ne pas saturer le modèle
        recent_memories = memories[-5:] # On prend les 5 derniers
        mem_texts = [f"- [{time.strftime('%Y-%m-%d', time.localtime(m['timestamp']))}] {m['text']}" for m in recent_memories]
        mem_section = "Souvenirs récents:\n" + "\n".join(mem_texts) + "\n\n"
    
    # Ajoute les instructions et les souvenirs au message utilisateur
    return f"{SYSTEM_PROMPT}\n\n{mem_section}Utilisateur: {user_msg}\nPhiFriend:"

def generate_reply(prompt):
    """Génère la réponse du modèle Phi-2."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_CFG)
    
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Tentative d'extraction de la réponse propre
    if prompt in decoded:
        reply = decoded.split(prompt, 1)[1].strip()
    else:
        # Fallback si le prompt n'est pas exactement retrouvé (moins précis)
        reply = decoded[len(decoded) // 2:].strip()
        
    return reply.split("\nUtilisateur:")[0].strip()

# -----------------------
# Fonctionnalité de Web Scraping
# -----------------------
def fetch_web_content(url):
    """Récupère et nettoie le texte principal d'une page web."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()

        text_content = soup.get_text()

        cleaned_text = re.sub(r'[\r\n]+', '\n', text_content) 
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()
        
        MAX_CHARS = 2500
        if len(cleaned_text) > MAX_CHARS:
             print(f"AVERTISSEMENT: Contenu tronqué à {MAX_CHARS} caractères.")
             return cleaned_text[:MAX_CHARS] + " [TRONQUÉ]..."
             
        return cleaned_text
        
    except requests.exceptions.MissingSchema:
        return "ERREUR: L'URL doit commencer par 'http://' ou 'https://'."
    except requests.exceptions.RequestException as e:
        return f"ERREUR de connexion/réseau : {e}"
    except Exception as e:
        return f"ERREUR inconnue : {e}"

# -----------------------
# Fonction d'API (Le pont vers Flask)
# -----------------------
def auto_store(user_text):
    """Vérifie si le message doit être enregistré en mémoire."""
    low = user_text.lower()
    triggers = ["souviens-toi", "souviens toi", "confie", "je me confie", "important", "anniversaire"]
    if any(t in low for t in triggers):
        remember(user_text, "confession" if "confie" in low else "event")
        return True
    return False

def process_user_message(user_input):
    """Fonction principale appelée par l'API Flask."""
    
    # --- Gestion de l'analyse d'URL ---
    if user_input.lower().startswith("analyse url:"):
        url = user_input[len("analyse url:"):].strip()
        print(f">>> Récupération et nettoyage du contenu de : {url}...")
        
        web_text = fetch_web_content(url)
        
        if web_text.startswith("ERREUR"):
            return web_text # Retourne l'erreur directement
        
        # Le prompt pour l'analyse web
        user_msg = (
            f"L'utilisateur souhaite une analyse. Le texte suivant provient de {url}. "
            f"Lis-le et donne une réponse concise, amicale et utile. "
            f"TEXTE DE LA PAGE:\n---\n{web_text}\n---"
        )
    
    else:
        # Message de chat standard
        user_msg = user_input
        # Stocker en mémoire si nécessaire
        if auto_store(user_msg):
            print(">> J’ai enregistré ça si tu veux que je m’en souvienne.")
            
    # Générer la réponse
    prompt = build_prompt(user_msg)
    reply = generate_reply(prompt)
    
    return reply

# NOTE: La boucle 'chat_loop' CLI a été supprimée.