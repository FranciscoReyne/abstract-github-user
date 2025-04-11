# abstract-github-user
Resumidor de usuarios de github con AI

**ATENCION: proyecto en contruccion.** Requiere google api key : https://aistudio.google.com/app/apikey

````python

# Instalamos las dependencias necesarias
!pip install -q langchain langchain_google_genai google-generativeai tqdm

import os
import requests
import base64
import time
from tqdm import tqdm
import re
from langchain_google_genai import GoogleGenerativeAIChat  # Corregido
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def get_user_repos(username):
    """Obtiene todos los repositorios de un usuario sin token."""
    repos = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/users/{username}/repos?page={page}&per_page=100'
        )
        if response.status_code != 200:
            print(f"Error al obtener repositorios: {response.status_code}")
            if response.status_code == 403:
                print("Límite de tasa de GitHub alcanzado. Espera una hora e intenta de nuevo.")
            break
        
        page_repos = response.json()
        if not page_repos:
            break
        
        repos.extend(page_repos)
        page += 1
        # Pausa para respetar los límites de tasa de GitHub
        time.sleep(2)
    
    return repos

def get_repo_contents(repo_full_name, path=''):
    """Obtiene el contenido de un repositorio recursivamente sin token."""
    response = requests.get(
        f'https://api.github.com/repos/{repo_full_name}/contents/{path}'
    )
    
    if response.status_code != 200:
        if response.status_code == 403:
            print("Límite de tasa de GitHub alcanzado. Espera una hora e intenta de nuevo.")
        return []
    
    contents = response.json()
    if not isinstance(contents, list):
        contents = [contents]
    
    # Pausa para respetar los límites de tasa
    time.sleep(1)
    return contents

def get_file_content(file_url):
    """Obtiene el contenido de un archivo sin token."""
    response = requests.get(file_url)
    if response.status_code != 200:
        return ""
    
    content = response.json()
    if 'content' in content and content['encoding'] == 'base64':
        try:
            decoded_content = base64.b64decode(content['content']).decode('utf-8')
            return decoded_content
        except:
            return "Archivo binario o con errores de codificación"
    return ""

def is_important_file(file_name):
    """Determina si un archivo es importante para el análisis."""
    # Extensiones de interés
    important_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.h', '.html', '.css', 
                          '.md', '.txt', '.json', '.yml', '.yaml', '.sh', '.rb', '.go']
    
    # Archivos importantes específicos
    important_files = ['README.md', 'requirements.txt', 'package.json', 'Dockerfile', '.gitignore']
    
    file_ext = os.path.splitext(file_name)[1].lower()
    return file_ext in important_extensions or file_name in important_files

def sample_repo_content(repo_full_name, max_files=5, max_content_length=3000):
    """Muestrea algunos archivos del repositorio para análisis."""
    all_contents = []
    dirs_to_explore = ['']
    files_sampled = 0
    
    while dirs_to_explore and files_sampled < max_files:
        current_dir = dirs_to_explore.pop(0)
        try:
            contents = get_repo_contents(repo_full_name, current_dir)
        except Exception as e:
            print(f"Error al explorar {current_dir}: {str(e)}")
            continue
        
        # Primero priorizar README.md
        readme_items = [item for item in contents if item['type'] == 'file' and item['name'].lower() == 'readme.md']
        if readme_items and files_sampled < max_files:
            for item in readme_items:
                content = get_file_content(item['url'])
                if content:
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "\n... [contenido truncado]"
                    all_contents.append({
                        'name': item['name'],
                        'path': item['path'],
                        'content': content
                    })
                    files_sampled += 1
        
        # Luego otros archivos importantes
        for item in contents:
            if files_sampled >= max_files:
                break
                
            if item['type'] == 'dir':
                dirs_to_explore.append(item['path'])
            elif item['type'] == 'file' and item['name'].lower() != 'readme.md':
                if is_important_file(item['name']):
                    content = get_file_content(item['url'])
                    if content:
                        if len(content) > max_content_length:
                            content = content[:max_content_length] + "\n... [contenido truncado]"
                        all_contents.append({
                            'name': item['name'],
                            'path': item['path'],
                            'content': content
                        })
                        files_sampled += 1
                        # Pausa para respetar los límites de tasa
                        time.sleep(1)
    
    return all_contents

def setup_llm():
    """Configura el modelo de lenguaje (Gemini Pro de Google)."""
    # Configurar la clave API para Gemini
    from google.colab import userdata
    
    # Intentar obtener la clave API
    try:
        api_key = userdata.get('GOOGLE_API_KEY', '')
    except:
        api_key = ''
        
    if not api_key:
        print("Necesitamos una API key de Google para Gemini. ")
        print("Puedes obtener una gratis en: https://makersuite.google.com/app/apikey")
        api_key = input("Ingresa tu API key de Google Gemini: ")
        # Establecer la clave API
        os.environ["GOOGLE_API_KEY"] = api_key
        # Intentar guardar para futuras referencias
        try:
            userdata.set('GOOGLE_API_KEY', api_key)
        except:
            pass
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    # Crear modelo
    llm = GoogleGenerativeAIChat(
        model="gemini-pro",
        temperature=0.5
    )
    
    return llm

def analyze_repository(llm, repo_data, repo_contents):
    """Analiza un repositorio individual usando LLM."""
    
    # Crear una descripción del repositorio
    repo_description = f"""
    Nombre: {repo_data['name']}
    Descripción: {repo_data['description'] or 'Sin descripción'}
    Lenguaje principal: {repo_data['language'] or 'No especificado'}
    Estrellas: {repo_data['stargazers_count']}
    Forks: {repo_data['forks_count']}
    """
    
    # Crear un resumen de los archivos muestreados
    files_summary = ""
    for file in repo_contents:
        files_summary += f"\nArchivo: {file['path']}\n"
        content_preview = file['content'][:300].replace("```", "'''")
        files_summary += f"Contenido muestra: {content_preview}...\n"
    
    # Crear el prompt para el LLM
    prompt = PromptTemplate(
        input_variables=["repo_description", "files_summary"],
        template="""
        Analiza este repositorio de GitHub y proporciona un resumen conciso de lo que hace, 
        la complejidad del código, las tecnologías utilizadas y cualquier patrón o enfoque 
        de programación notable.
        
        Información del repositorio:
        {repo_description}
        
        Muestra de archivos:
        {files_summary}
        
        Resumen (máximo 150 palabras):
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.invoke({
            "repo_description": repo_description,
            "files_summary": files_summary
        })
        return result["text"].strip()
    except Exception as e:
        print(f"Error al analizar el repositorio: {str(e)}")
        return f"Error al analizar el repositorio: {str(e)}"

def create_user_profile(llm, username, repo_analyses):
    """Crea un perfil de usuario basado en los análisis de los repositorios."""
    
    # Crear un resumen de los análisis de repositorios
    repos_summary = "\n\n".join([
        f"Repositorio: {repo_name}\n{analysis}" 
        for repo_name, analysis in repo_analyses.items()
    ])
    
    # Crear el prompt para el LLM
    prompt = PromptTemplate(
        input_variables=["username", "repos_summary"],
        template="""
        Con base en los siguientes análisis de repositorios de GitHub del usuario {username}, 
        crea un perfil profesional resumido que destaque:
        
        1. Áreas de experiencia técnica
        2. Lenguajes y tecnologías dominantes
        3. Nivel de habilidad aproximado
        4. Intereses de desarrollo principales
        5. Fortalezas como desarrollador
        
        Análisis de repositorios:
        {repos_summary}
        
        Perfil profesional (máximo 300 palabras):
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.invoke({
            "username": username,
            "repos_summary": repos_summary
        })
        return result["text"].strip()
    except Exception as e:
        print(f"Error al crear el perfil de usuario: {str(e)}")
        return f"Error al crear el perfil de usuario: {str(e)}"

def analyze_github_profile(username, max_repos=5):
    """Función principal para analizar el perfil de GitHub de un usuario."""
    # Configurar el modelo
    llm = setup_llm()
    
    print(f"Obteniendo repositorios para {username}...")
    repos = get_user_repos(username)
    print(f"Se encontraron {len(repos)} repositorios")
    
    # Ordenar repos por popularidad (estrellas)
    repos_sorted = sorted(repos, key=lambda x: x['stargazers_count'], reverse=True)
    
    # Limitar el número de repos a analizar
    repos_to_analyze = repos_sorted[:max_repos]
    
    repo_analyses = {}
    
    for repo in tqdm(repos_to_analyze, desc="Analizando repositorios"):
        print(f"\nAnalizando: {repo['name']}")
        # Obtener contenido del repositorio
        repo_contents = sample_repo_content(repo['full_name'])
        # Analizar repositorio si se obtuvieron contenidos
        if repo_contents:
            analysis = analyze_repository(llm, repo, repo_contents)
            repo_analyses[repo['name']] = analysis
        else:
            print(f"No se pudo obtener contenido para {repo['name']}")
    
    # Crear perfil de usuario
    print("\nGenerando perfil de usuario...")
    user_profile = create_user_profile(llm, username, repo_analyses)
    
    return user_profile

# Ejemplo de uso
if __name__ == "__main__":
    username = input("Ingresa el nombre de usuario de GitHub: ")
    max_repos = int(input("Número máximo de repositorios a analizar (recomendado: 3-5): "))
    
    profile = analyze_github_profile(username, max_repos)
    
    print("\n" + "="*50)
    print(f"PERFIL DE {username.upper()}")
    print("="*50)
    print(profile)

````

# FIN.-

Saludos !! :D
