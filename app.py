import os
from flask import Flask, request, jsonify, render_template

# Importer la logique de l'IA que nous avons mise dans ia_core.py
import ia_core 

# Configuration de l'application Flask
app = Flask(__name__)

# Route pour la page d'accueil (le chat HTML)
@app.route('/')
def index():
    """Affiche la page HTML du chat depuis le dossier templates."""
    return render_template('index.html')

# Route pour l'API de discussion
@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Reçoit le message de l'utilisateur et renvoie la réponse de l'IA."""
    
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"reply": "Veuillez entrer un message."})

    try:
        # Appeler la fonction de traitement de l'IA
        ai_reply = ia_core.process_user_message(user_message)
        
        # Renvoyer la réponse au navigateur
        return jsonify({"reply": ai_reply})
    
    except Exception as e:
        # Gérer les erreurs de modèle ou autres
        print(f"Erreur lors du traitement : {e}")
        return jsonify({"reply": "Désolé, une erreur interne est survenue sur le serveur."}), 500

# Le bloc ci-dessous n'est utilisé que pour les tests locaux, Render utilise Gunicorn.
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Note: debug=False en production (Render) est la meilleure pratique
    app.run(debug=True, port=port)