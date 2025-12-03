from setuptools import setup, find_packages

setup(
    name="Biatron",               # Nome do seu pacote (como aparecerá no 'pip list')
    version="0.1.1",
    description="Biatron Model Implementation",
    
    # Encontra pacotes automaticamente (ex: 'meu_pacote' dentro de 'src')
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # ESSENCIAL: Defina as dependências
    install_requires=[
        "transformers>=4.0.0",
        "torch",
    ],
    
    author="Daniel Fazzioni",
    author_email="Fazzioni@discente.ufg.br"
)