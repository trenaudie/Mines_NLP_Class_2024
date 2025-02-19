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

response = get_response(f'donner un résumé de  cet arrêté préfectoral : {soup_text}')

# %%
print(response) 
# %%
response = """Voici un résumé de l'arrêté préfectoral :

Cet arrêté, daté du 3 août 1994, autorise administrativement l'usine COGESAL S.A d'Argentan, à exploiter ses activités industrielles, sous réserve du respect de nombreuses prescriptions techniques. Il prend en compte une demande de régularisation, une enquête publique favorable, et divers avis consultatifs.

L'arrêté encadre les activités suivantes :

*   Dépôt de liquides peu et extrêmement inflammables.
*   Installations de réfrigération.
*   Épandage d'effluents.

Il fixe des règles générales concernant :

*   La protection des eaux (normes de rejet, prévention des pollutions accidentelles, gestion des eaux pluviales et usées).
*   La prévention de la pollution atmosphérique (limitation des émissions de fumées, gaz etc.).
*   L'élimination des déchets.
*   Les nuisances sonores (conformité aux normes).
*   Les installations électriques (sécurité et contrôle).
*   Les règles de circulation sur le site.
*   La prévention des risques d'incendie et d'explosion (détection, moyens de secours, interdiction de fumer, permis de feu).
*   Activités laitières et de fabrication de produits laitiers.
*   Installations de réfrigération fonctionnant à l'ammoniac.
*   Installations de Combustion.

L'arrêté précise également les échéances pour certaines études ou mises en conformité, les modalités de contrôle, et les obligations de publicité de la décision. Il abroge plusieurs arrêtés et réceptions de déclaration antérieures.
"""
final_response = get_response(f'translate to english : {response}')
# %%
print(final_response)
# %%
