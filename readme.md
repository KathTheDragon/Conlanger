# Conlanger

Conlanger is a package containing several tools designed primarily to aid conlangers with tedious tasks such as word
generation and diachronics. To use, simply place the package on your Python path, and `import conlanger` into an
interface script.

## Gen

`conlanger.gen` is a module allowing the generation of words from syllables whose graphemes, where the syllable types
and graphemes are both distributed according to peaked power law distributions. Additionally, restrictions (both linear
and non-linear) can be placed on what outputs are considered valid.

## SCE

`conlanger.sce` is a module with powerful tools for transforming words according to transformation rules. Documentation
of the rules can be found [here](https://conworkshop.com/leashy/sce/sce-doc.html).

## Lang

`conlanger.lang` is a module providing support for storing the configuration data for the other modules on a
per-language basis, as well as saving this data to and loading from file. It also provides shortcuts to utilising the
other modules with a given language, automatically providing the configuation data defined for that language.
