# Artificial Cambrian Intelligence (ACI)

This is the repo/website for the paper

**[A Roadmap for Generative Design of Visual Intelligence](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3)** \
[Kushagra Tiwary](http://kushagratiwary.com/), [Tzofi Klinghoffer\*](https://tzofi.github.io/), [Aaron Young\*](https://AaronYoung5.github.io/), [Siddharth Somasundaram](https://sidsoma.github.io/), [Nikhil Behari](https://nikhilbehari.github.io/), [Akshat Dave](https://akshatdave.github.io/), [Brian Cheung](https://briancheung.github.io/), [Dan-Eric Nilsson](https://portal.research.lu.se/en/persons/dan-eric-nilsson), [Tomaso Poggio](https://mcgovern.mit.edu/profile/tomaso-poggio/), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/)


[Paper](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3) | [Code](https://github.com/camera-culture/ACI) | [Documentation](https://camera-culture.github.io/ACI/)

The incredible diversity of visual systems in the animal kingdom is a result of millions of years of coevolution between eyes and brains, adapting to process visual information efficiently in different environments. We introduce the generative design of visual intelligence (GenVI), which leverages computational methods and generative artificial intelligence to explore a vast design space of potential visual systems and cognitive capabilities. By co-generating artificial eyes and brains that can sense, perceive, and enable interaction with the environment, GenVI enables the study of the evolutionary progression of vision in nature and the development of novel and efficient artificial visual systems. We anticipate that GenVI will provide a powerful tool for vision scientists to test hypotheses and gain new insights into the evolution of visual intelligence while also enabling engineers to create unconventional, task-specific artificial vision systems that rival their biological counterparts in terms of performance and efficiency.

## Installation

This page provides setup and installation information for the `ACI` repo. Python >= 3.11 is required.

First, clone the repo:

```
git clone https://github.com/camera-culture/ACI.git
```

Then you can install the `cambrian` package by doing the following. Note that the package was actually designed to be used with [poetry](https://python-poetry.org/docs/), and is required when [contributing](https://camera-culture.github.io/ACI/contributing.html) to the project. 


```bash
pip install -e .
```

## Usage

To simply train a single agent in a detection-style task, you can run the following:

```bash
bash scripts/local.sh scripts/train.sh exp=tasks/detection
```

For more information on how to train, evaluate, and run experiments, see the [Documentation](https://camera-culture.github.io/ACI/usage/index.html) website.

## Citation

```bibtex
@article{Tiwary2024Roadmap,
	author = {Tiwary, Kushagra and Klinghoffer, Tzofi and Young, Aaron and Somasundaram, Siddharth and Behari, Nikhil and Dave, Akshat and Cheung, Brian and Nilsson, Dan-Eric and Poggio, Tomaso and Raskar, Ramesh},
	journal = {An MIT Exploration of Generative AI},
	year = {2024},
	month = {sep 18},
	note = {https://mit-genai.pubpub.org/pub/bcfcb6lu},
	publisher = {MIT},
	title = {A {Roadmap} for {Generative} {Design} of {Visual} {Intelligence}},
}
```
