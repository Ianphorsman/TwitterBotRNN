from setuptools import setup

setup(
    name='real_fake_trump_bot',
    description="Tooling resource for analyzing and emulating donald trump's tweets",
    url="https://github.com/Ianphorsman/TrumpBot",
    author='Ian Horsman',
    author_email='ianphorsman@gmail.com',
    license='MIT',
    version='0.1.0',
    packages=['real_fake_trump_bot'],
    install_requires=['numpy', 'nltk', 'scikit-learn', 'seaborn', 'tensorflow-gpu', 'keras']
)