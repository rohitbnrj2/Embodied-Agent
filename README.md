# Artificial Cambrian Intelligence (ACI)

This is the repo/website for the paper

**[A Roadmap for Generative Design of Visual Intelligence](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3)** \
[Kushagra Tiwary](http://kushagratiwary.com/), [Tzofi Klinghoffer\*](https://tzofi.github.io/), [Aaron Young\*](https://AaronYoung5.github.io/), [Siddharth Somasundaram](https://sidsoma.github.io/), [Nikhil Behari](https://nikhilbehari.github.io/), [Akshat Dave](https://akshatdave.github.io/), [Brian Cheung](https://briancheung.github.io/), [Dan-Eric Nilsson](https://portal.research.lu.se/en/persons/dan-eric-nilsson), [Tomaso Poggio](https://mcgovern.mit.edu/profile/tomaso-poggio/), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/)

[Paper](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3) | [Code](https://github.com/camera-culture/ACI) | [Documentation](https://camera-culture.github.io/ACI/)

The incredible diversity of visual systems in the animal kingdom is a result of millions of years of coevolution between eyes and brains, adapting to process visual information efficiently in different environments. We introduce the generative design of visual intelligence (GenVI), which leverages computational methods and generative artificial intelligence to explore a vast design space of potential visual systems and cognitive capabilities. By co-generating artificial eyes and brains that can sense, perceive, and enable interaction with the environment, GenVI enables the study of the evolutionary progression of vision in nature and the development of novel and efficient artificial visual systems. We anticipate that GenVI will provide a powerful tool for vision scientists to test hypotheses and gain new insights into the evolution of visual intelligence while also enabling engineers to create unconventional, task-specific artificial vision systems that rival their biological counterparts in terms of performance and efficiency.

## Installation

This page provides setup and installation information for the `ACI` repo. Python >= 3.11 is required.

First, clone the repo:

```bash
git clone https://github.com/camera-culture/ACI.git
```

Then you can install the `cambrian` package by doing the following.

```bash
pip install -e .
```

## Usage

### Test

To test the setup and verify you can visualize the environment, you can run the following:

```bash
# Setting frame_skip slows down the agent's movements to make it easier to see, the default is 10.
python cambrian/main.py --eval example=detection_optimal env.renderer.render_modes='[human]' env.frame_skip=5
```

This command should open a window showing an agent moving towards a target. It uses an "optimal" policy which just tries to minimize the distance to the target always.

Currently, the available examples are:

- `light_seeking`: An agent moving towards a single light source.
- `navigation`: A single agent navigating a large maze.
- `navigation_ma`: `navigation`, but with multiple agents.
- `detection`: A single agent moving towards a target while avoiding an obstacle.
- `detection_ma`: `detection`, but with multiple agents.
- `tracking`: Similar to `detection`, but the target and obstacle move.

Any of the commands which show `example=<example>` can use any of the above examples, including `<example>_optimal` for the optimal policy.

### Train

To train a single agent in a detection-style task, you can run the following command. You will find the trained model and output files at `log/<today's date>/exp_detection`. This should take like 10 minutes to an hour depending on your computer.

```bash
bash scripts/run.sh cambrian/main.py --train example=detection
```

After training, you can evaluate the agent using the following:

```bash
python cambrian/main.py --eval example=detection env.renderer.render_modes='[human]' trainer/model=loaded_model
```

## Documentation

For more detailed information on how to train, evaluate, and run experiments, see the [Documentation](https://camera-culture.github.io/ACI/usage/index.html) website.

### Compiling the Documentation

First install the dev/doc dependencies.

```bash
pip install -e '.[doc,dev]'
```

Then to install, run the following:

```bash
cd docs
make clean html
```

To view the build, go to your browser, and open the `index.html` file located inside `docs/build/html/`.

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
