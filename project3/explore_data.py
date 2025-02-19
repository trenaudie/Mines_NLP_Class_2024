#%% 
from pathlib import Path
DATA_DIR = Path("project3/data")
file = DATA_DIR.iterdir().__next__()
str(file)
# %%
len(list(DATA_DIR.iterdir()))
# %%
from project3.get_response import get_response

file = "1994-08-03_AP-auto_refonte_pixtral.html"
file = DATA_DIR / file
file_text= open(file, "r").read()
# %%
file_text
from bs4 import BeautifulSoup
soup = BeautifulSoup(file_text, "html.parser")
# %%
soup_text = soup.get_text()
# %%
soup_text
# %%

response = get_response(f'expliquer la structure de cet arrêté prefectoral : {soup_text}')

# %%
print(response) 
# %%
response = """VL'arrêté préfectoral que vous avez fourni est un document juridique détaillé qui autorise l'exploitation d'une installation classée pour la protection de l'environnement (ICPE). Voici une explication de sa structure :

**1. En-tête : Identification du Document et de l'Autorité**

*   **SERVICE DES ACTIONS DE L'ETAT** : Indique le service administratif de la préfecture responsable de l'arrêté.
*   **Préfecture de l'Orne le [Date]** : Précise l'administration émettrice (la préfecture de l'Orne) et la date de l'arrêté.
*   **Bureau INSTALLATION CLASSEE SOUMISE A AUTORISATION Commune d'ARGENTAN Usine COGESAL S.A.** : Localise l'installation concernée (commune d'Argentan, usine Cogesal S.A.) et précise que l'installation est soumise à autorisation au titre des ICPE.
*   **ARRETE D'AUTORISATION** : Déclare clairement le type de document.

**2. Vus (Motifs Juridiques et Procéduraux)**

Cette section liste toutes les lois, décrets, avis et consultations qui ont servi de base légale à la décision préfectorale. Chaque élément cité a une importance juridique :

*   **Lois et Décrets Généraux :** Ils établissent le cadre légal pour les ICPE et la protection de l'environnement (ex : loi de 1976 sur les ICPE, loi sur l'eau de 1992).
*   **Demande de l'Exploitant :** Mentionne la demande d'autorisation soumise par le Directeur de l'usine COGESAL S.A., précisant l'objet de la demande (régularisation administrative).
*   **Dossier Technique :** Fait référence aux plans et documents techniques inclus dans le dossier de demande d'autorisation.
*   **Enquête Publique :** Indique qu'une enquête publique a été menée (dates) et fait référence à l'avis du commissaire-enquêteur.
*   **Avis des Collectivités et Services :** Liste les avis émis par les conseils municipaux des communes concernées, ainsi que les différents services départementaux (DASS, DDE, SDIS, etc.)
*   **Rapports Techniques :** Mentionne le rapport du DRIRE (direction régionale de l'industrie, de la recherche et de l'environnement) et l'avis du Conseil Départemental d'Hygiène (CDH).
*   **Arrêtés de Sursis à Statuer :** Indique que la décision a été suspendue temporairement.
*   **Considérant :** Explique que le projet d'arrêté a été communiqué au demandeur.

**En résumé, cette section démontre que la préfecture a suivi toutes les procédures légales et consulté toutes les parties prenantes avant de prendre sa décision.**

**3. Parties centrales du document, classification des activités de l'établissement :**

Cette section est importante car elle décrit les activités de l'établissement soumises à autorisation.
*   **Rubrique de la nomenclature ICPE :** Chaque activité est identifiée par un numéro de rubrique de la nomenclature des installations classées (ex: 1430 - D). Ces numéros correspondent à des types d'installations ou d'activités spécifiques.
*   **Désignation des activités :** Décrit la nature de l'activité (ex : "Dépôt de liquides peu inflammables").
*   **Régime (A/D):**  Indique le régime administratif applicable à l'activité : A pour Autorisation (soumise aux prescriptions détaillées de l'arrêté), D pour Déclaration (soumise à des prescriptions générales définies dans les arrêtés ministériels types).
*   **Caractéristiques :** Précise les seuils ou les quantités qui rendent l'activité soumise à autorisation (ex : capacité de stockage, puissance installée...).

**4. Articles Principaux (Prescriptions et Obligations)**

Cette section contient les articles qui définissent les obligations de l'exploitant. On peut la diviser en plusieurs parties :

*   **Article 2 :** Annonce que les activités soumises à autorisation doivent respecter les prescriptions techniques qui suivent. Abroge les arrêtés antérieurs obsolètes.
*   **I - DISPOSITIONS GENERALES :**

    *   **Article 3 :** Précise que l'autorisation ne dispense pas l'exploitant de respecter les autres réglementations (urbanisme, santé, travail...). Réserve les droits des tiers.
    *   **Article 4 :** Demande de réaliser les installations conformément aux plans, d'utiliser les meilleures technologies disponibles, et de soumettre toute modification notable à l'approbation de l'inspection des ICPE.
    *   **Article 5 :** Clôture du site, propreté et entretien.
    *   **Article 6 :** Prévention des pollutions accidentelles et obligation d'informer l'inspection des ICPE en cas d'incident.
    *   **Article 7 :** Droit de l'inspection des ICPE de réaliser des contrôles et analyses aux frais de l'exploitant.
    *   **Article 8 :** Caducité de l'autorisation en cas d'arrêt de l'exploitation pendant deux ans.
*   **II - PRESCRIPTIONS TECHNIQUES GENERALES :**

    *   Cette partie détaille les prescriptions techniques à respecter dans différents domaines :
        *   **A - PROTECTION DES EAUX :** Normes de rejet, prévention des pollutions accidentelles, gestion des eaux pluviales et usées, etc. Article 9
        *   **B - PREVENTION DE LA POLLUTION ATMOSPHERIQUE :** Collecte et traitement des rejets atmosphériques, interdiction des fumées incommodantes, interdiction du brûlage à l'air libre. Article 10
        *   **C - ELIMINATION DES DECHETS :** Stockage et élimination des déchets dans des conditions respectueuses de l'environnement, traçabilité des déchets. Article 11
        *   **Article 12 : élimination des boues de la station d'épuration. Conditions de traitement, règles d'épandage, bilan annuel et suivi agronomique.
        *   **D - PREVENTION DES NUISANCES SONORES :** Limitation des bruits et vibrations, respect de l'arrêté ministériel du 20 août 1985. Articles 13 à 16
        *   **E - INSTALLATIONS ELECTRIQUES :** Conformité aux normes, contrôles périodiques, protection contre la foudre. Articles 17 et 18
        *   **F - REGLES DE CIRCULATION :** Règles de circulation à l'intérieur du site, signalisation, accès pour les secours. Article 19
        *   **G - PREVENTION ET PROTECTION CONTRE LES RISQUES D'INCENDIE ET D'EXPLOSION :** Conception des bâtiments, moyens de détection et de secours, interdiction de fumer, permis de feu, consignes générales de sécurité et d'incendie. Articles 20 à 24
    *   **III - PRESCRIPTIONS COMPLEMENTAIRES PARTICULIERES APPLICABLES AUX ATELIERS DE TRAVAIL DU LAIT ET DERIVES** Décrit le processus de fabrication autorisé et les quantités maximales de produits traitées par jour. Articles 25 et 26
    *   **IV - PRESCRIPTIONS COMPLEMENTAIRES PARTICULIERES APPLICABLES AUX INSTALLATIONS DE REFRIGERATION FONCTIONNANT A L'AMMONIAC** Liste les limites et contraintes à respecter pour l'utilisation des installations de réfrigération fonctionnant à l'ammoniac. Articles 27 à 30
    *   **V - PRESCRIPTIONS COMPLEMENTAIRES PARTICULIERES APPLICABLES AUX INSTALLATIONS DE COMBUSTION** Décrit les caractéristiques obligatoires des deux chaudières. Article 31

*   **V - RAPPEL DES ECHEANCES**
    *   **Article 32 :** Rappel des dates butoirs de réalisation de certaines prescriptions spécifiées dans les articles précédents.

**5. Dispositions Finales (Notification, Publicité, Ampliation)**

*   **Article 33 :** Affichage de l'arrêté à la mairie, insertion d'un avis dans les journaux.
*   **Article 34 :** Indique les personnes et services chargés de l'exécution de l'arrêté et à qui il doit être notifié ou amplifié (c'est-à-dire transmis pour information).

**6. Signature**

*   Le document est signé par le Préfet (ou par délégation), attestant de la validité de l'arrêté.

**En conclusion, cet arrêté préfectoral est un document complexe qui vise à encadrer l'exploitation d'une ICPE afin de protéger l'environnement et la santé publique. Il est essentiel que l'exploitant en comprenne toutes les dispositions et les respecte scrupuleusement.**"""
final_response = get_response(f'translate to english : {response}')
# %%
print(final_response)
# %%
from pathlib import Path

# select first file
file = list(DATA_DIR.iterdir())[0]

# read file
with open(file, "r") as f:
    text = f.read()
print(text)
soup = BeautifulSoup(text, "html.parser")
# %%
textgotten = soup.get_text()
# %%
print(textgotten)
# %%
# remove the "Vu" lines 
filtering_prompt = """
You are a text extraction expert.
You are given a text.
Your job is to extract the text that is relevant to the classification of the prefectoral order.
The text is a prefectoral order.

It will have a structure like this:

**1. Header: Document and Authority Identification**

**2. Vus (Legal and Procedural Grounds)**

This section demonstrates that the prefecture followed all legal procedures and consulted all stakeholders before making its decision.

**3. Central Parts of the Document, Classification of the Establishment's Activities:**

This section is important because it describes the activities of the establishment that are subject to authorization.

**4. Main Articles (Prescriptions and Obligations)**

This section contains the articles that define the operator's obligations. It can be divided into several parts:

**5. Final Provisions (Notification, Publicity, Copies)**

**6. Signature**

*   The document is signed by the Prefect (or by delegation), attesting to the validity of the order.

You will extract the parts 1 and 3, ie the header and the central parts of the document and classification of the establishment's activities.

You will also extract the number of the prefectoral order if it is given, its date if it is given. 

You will return the text in a json format with the following keys:
- number: the number of the prefectoral order, if it is given, otherwise None. It may be given as a NOR number at the top of the document. It is a number in the following format NOR : 1234-56-78912
- date: the date of the prefectoral order, if it is given, otherwise None
- header: the header of the prefectoral order
- central_parts: the central parts of the document and classification of the establishment's activities.

You must not modify the text from the prefectoral order in any way, simply extract the information.
Here is the prefectoral order text:
"""
#%%
full_prompt = f"{filtering_prompt}\n\n{textgotten}"

# %%
response = get_response(full_prompt)
# %%
import json 
def extract_json(response:str):
    cleaned = response.replace('```json', '').replace('```', '')
    cleaned = cleaned.strip()
    return json.loads(cleaned)


# %%
obj = extract_json(response)
obj
# %%
print(obj["header"])
# %%
print(obj["central_parts"])
# %%
classification_prompt = """
You are a text classification agent. 
You are given a prefectoral order, in the form of a json object like this:
{
    "number": "1234567890",
    "date": "2024-01-01",
    "header": "Header of the prefectoral order",
    "central_parts": "Central parts of the prefectoral order"
}
Your job is to classify the prefectoral order based on its action. 
Here are some possible actions: 
- "authorize action" : the prefectoral order authorizes an activity from an establishment
- "replace or modify order" : the prefectoral order replaces or modifies an existing order
- "close or stop activity" : the prefectoral order closes or stops an activity from an establishment
You can also return a different action, if it is not one of the above.

You will return the classification output in the following json format:
```json
{

"arrete prefectoral n° 2350-22-00082 du 12 mai 2022": {
    "label": "authorize action",
    "target": "S.A. ROXANE",  #target of the action
    "establishment" : S.A ROXANE" #company being affected by the prefectoral order
}

}
```
where the key of the json object is a string representing the prefectoral order, in French, with the format
"arrêté préfectoral n° [number] du [date]" with the number if it is given and the date if it is given.

If the label is "replace or modify order", the target is the order, prescription or text that is being modified or replaced.

Here is the prefectoral order json object:
"""
#%%
full_prompt = f"{classification_prompt}\n\n{obj}"

# %%
response = get_response(full_prompt)
# %%
print(response)
# %%

# %%
import json

def clean_and_parse_json(unclean_string:str):
    import re
    cleaned_string = unclean_string.strip().strip("'").replace('\\n', '\n').replace('\\"', '"')

    cleaned_string = cleaned_string.replace('```json', '').replace('```', '').strip()

    cleaned_string = cleaned_string.replace('\\n', '\n').replace('\\"', '"')    

    cleaned_string = re.sub(r',\s*}', '}', cleaned_string)
    try:
        python_object = json.loads(cleaned_string)
        return python_object
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

clean_string = clean_and_parse_json(response)
# %%
RESULTS_DIR = Path("project3/results")
RESULTS_DIR.mkdir(exist_ok=True)
all_responses = dict()
for file in DATA_DIR.iterdir():
    with open(file, "r") as f:
        text = f.read()
    soup = BeautifulSoup(text, "html.parser")
    textgotten = soup.get_text()
    full_filtered_prompt = f"{filtering_prompt}\n\n{textgotten}"
    response = get_response(full_filtered_prompt)
    obj = extract_json(response)
    full_classification_prompt = f"{classification_prompt}\n\n{obj}"
    response = get_response(full_classification_prompt)
    print(response)
    clean_response = clean_and_parse_json(response)
    all_responses[file.name] = clean_response
    with open(RESULTS_DIR / f"{file.name}_result.json", "w") as f:
        json.dump(clean_response, f)
