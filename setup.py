from setuptools import setup

setup(
    name='colibri_hdsp',
    install_requires=['torch', 'torchmetrics', 'tqdm', 'torchvision', 'matplotlib'],
    extras_require={'doc': ['sphinx', 'furo', 'autodocsumm', 'sphinx_gallery', ]},
)