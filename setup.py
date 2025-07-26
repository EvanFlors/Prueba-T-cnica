from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path):
  '''
    Return the list of requirements
    
    Args:
      file_path (str): path to requirements.txt file
      
    Returns:
      List[str]: list of requirements
  '''
  with open(file_path) as file:
    requirements = file.readlines()
    requirements = [req.replace("\n", "") for req in requirements]
    
    if HYPEN_E_DOT in requirements:
      requirements.remove(HYPEN_E_DOT)
    
  return requirements

setup(
  name = 'prueba_tecnica',
  version = '0.1.0',
  author = 'Esteban',
  author_email = 'e2002florespulido@gmail.com',
  packages = find_packages(),
  install_requires = get_requirements('requirements.txt'),
  description = 'Prueba Tecnica',
  license = 'MIT',
)