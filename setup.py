from setuptools import setup, find_packages, Command, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils import log
from distutils.util import convert_path

setup_requires = [
    "Pillow"
]

extras_require = {
    "fonttools": []
}

setup_params = dict(
    name="textoverlaydataset",
    version="0.1.1",
    description="A tool to generate synthetic datasets from text and image datasets.",
    author="Joseph Catrambone",
    author_email="me@josephcatrambone.com",
    maintainer="Joseph Catrambone",
    maintainer_email="me@josephcatrambone.com",
    url="http://github.com/JosephCatrambone/PyTorchTextOverlayDataset",
    license="MIT",
    platforms=["Any"],
    python_requires=">=3.7",
    package_dir={"": "Lib"},
    packages=find_packages("Lib"),
    include_package_data=True,
    setup_requires=setup_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "fonttools = fontTools.__main__:main",
            "ttx = fontTools.ttx:main",
            "pyftsubset = fontTools.subset:main",
            "pyftmerge = fontTools.merge:main",
        ]
    },
    cmdclass=cmdclass,
    **classifiers,
)


if __name__ == "__main__":
    setup(**setup_params)