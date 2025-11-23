# Utilise l'image Python officielle, légère et stable
FROM python:3.10-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie le fichier requirements.txt et installe toutes les dépendances
# --no-cache-dir permet d'économiser de l'espace disque
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du code de l'application (app.py, ia_core.py, templates/, etc.)
COPY . .

# Expose le port 8000 (port par défaut pour Gunicorn dans un conteneur)
EXPOSE 8000

# Commande pour démarrer le serveur Gunicorn
# Le serveur se lie à l'adresse 0.0.0.0 sur le port 8000 et exécute l'application Flask nommée 'app' dans le fichier 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]