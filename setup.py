"""Package local model implementations alongside GenIA, including in wheels."""
from setuptools import find_namespace_packages, setup

models = ["sam3d_objects", "mapanything", "moge", "pytorch3d", "utils3d", "nvdiffrast", "diso", "lpips", "uniception", "vox2seq", "gsconverter", "actionmesh", "triposg", "pytorch_msssim"]
packages = ["genia.core"] + ["genia.core." + p for p in find_namespace_packages("core", exclude=["*__pycache__*"])]
mapping = {"genia": "."}
for name in models:
    packages += [name] + [name + "." + p for p in find_namespace_packages("vendor/" + name, exclude=["*__pycache__*"])]
    mapping[name] = "vendor/" + name
setup(packages=packages, package_dir=mapping)
