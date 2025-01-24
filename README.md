# Artificial Cambrian Intelligence (ACI)

Computational methods unveil intricate processes of visual system evolution by simulating environments where embodied agents simultaneously develop eye morphologies and neural processing. By exploring how task demands and environmental pressures shape vision, researchers demonstrate that different challenges trigger unique eye developments—navigation prompting compound eyes, object discrimination yielding camera-like structures—while revealing natural optical innovation emergence and systematic connections between visual acuity and neural complexity, ultimately providing a robust framework for understanding biological vision evolution and designing task-specific artificial visual systems.

## Project Papers

<div style="display: flex; align-items: center; gap: 1rem;">

<img src="https://eyes.mit.edu/ACI/_static/whatifeye.png" alt="What if Eye...?" width="100">

***What if Eye...?* Computationally Recreating Vision Evolution** \
[Kushagra Tiwary\*](https://kushagratiwary.com/), [Aaron Young\*](https://AaronYoung5.github.io/), [Zaid Tasneem](https://zaidtas.github.io/), [Tzofi Klinghoffer](https://tzofi.github.io/), [Akshat Dave](https://akshatdave.github.io/), [Tomaso Poggio](https://mcgovern.mit.edu/profile/tomaso-poggio/), [Dan-Eric Nilsson](https://portal.research.lu.se/en/persons/dan-eric-nilsson), [Brian Cheung<sup>†</sup>](https://briancheung.github.io/), [Ramesh Raskar<sup>†</sup>](https://www.media.mit.edu/people/raskar/overview/)
<br>
\[[Paper](https://google.com) | [Website](https://eyes.mit.edu) | [Code](https://github.com/cambrian-org/ACI) | [Documentation](https://eyes.mit.edu/ACI/)\]

<!-- > Vision systems in nature show remarkable diversity, from simple light-sensitive patches to complex camera eyes with lenses. While natural selection has produced these eyes through countless mutations over millions of years, they represent just one set of realized evolutionary paths. Testing hypotheses about how environmental pressures shaped eye evolution remains challenging since we cannot experimentally isolate individual factors. Computational evolution offers a way to systematically explore alternative trajectories. Here we show how environmental demands drive three fundamental aspects of visual evolution through an artificial evolution framework that co-evolves both physical eye structure and neural processing in embodied agents. First, we demonstrate that task demands bifurcate eye evolution -- navigation tasks lead to distributed compound-type eyes while object discrimination drives the emergence of high-acuity camera eyes. Second, we reveal how optical innovations like lenses naturally emerge to resolve fundamental tradeoffs between light collection and spatial precision. Third, we uncover systematic scaling laws between visual acuity and neural processing, showing how task complexity drives coordinated evolution of sensory and computational capabilities. Our work introduces a novel paradigm that illuminates evolutionary principles shaping vision by creating targeted single-player games where embodied agents must simultaneously evolve visual systems and learn complex behaviors. Through our unified genetic encoding framework, these embodied agents serve as next-generation hypothesis testing machines while providing a foundation for designing manufacturable bio-inspired vision systems. -->


</div>

<div style="display: flex; align-items: center; gap: 1rem;">

<img src="https://eyes.mit.edu/ACI/_static/genvi.png" alt="GenVI" width="100">

**A Roadmap for Generative Design of Visual Intelligence** \
[Kushagra Tiwary](https://kushagratiwary.com/), [Tzofi Klinghoffer\*](https://tzofi.github.io/), [Aaron Young\*](https://AaronYoung5.github.io/), [Siddharth Somasundaram](https://sidsoma.github.io/), [Nikhil Behari](https://nikhilbehari.github.io/), [Akshat Dave](https://akshatdave.github.io/), [Brian Cheung](https://briancheung.github.io/), [Dan-Eric Nilsson](https://portal.research.lu.se/en/persons/dan-eric-nilsson), [Tomaso Poggio](https://mcgovern.mit.edu/profile/tomaso-poggio/), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/)
<br>
\[[Paper](https://mit-genai.pubpub.org/pub/bcfcb6lu/release/3) | [Code](https://github.com/cambrian-org/ACI) | [Documentation](https://eyes.mit.edu/ACI/)\]

<!-- > The incredible diversity of visual systems in the animal kingdom is a result of millions of years of coevolution between eyes and brains, adapting to process visual information efficiently in different environments. We introduce the generative design of visual intelligence (GenVI), which leverages computational methods and generative artificial intelligence to explore a vast design space of potential visual systems and cognitive capabilities. By co-generating artificial eyes and brains that can sense, perceive, and enable interaction with the environment, GenVI enables the study of the evolutionary progression of vision in nature and the development of novel and efficient artificial visual systems. We anticipate that GenVI will provide a powerful tool for vision scientists to test hypotheses and gain new insights into the evolution of visual intelligence while also enabling engineers to create unconventional, task-specific artificial vision systems that rival their biological counterparts in terms of performance and efficiency. -->

</div>

## Setup

### Prerequisites

- Python 3.11 or higher

### Installation

First, clone the repo:

```bash
git clone https://github.com/cambrian-org/ACI.git
```

Then you can install the [`cambrian`](https://eyes.mit.edu/ACI/reference/api/cambrian/#module-cambrian) package by doing the following.

```bash
pip install -e .
```

## Usage

### Test

To test the setup and verify you can visualize the environment, you can run the following:

```bash
# Setting frame_skip slows down the agent's movements to make it easier to see, the default is 10.
python cambrian/main.py --eval example=detection env.renderer.render_modes='[human]' env.frame_skip=5 env/agents@env.agents.agent=point_seeker
```

This command should open a window showing an agent moving towards a target. It uses a privileged policy which just tries to minimize the distance to the target.

Currently, the available examples are:

- [Navigation Task](https://eyes.mit.edu/ACI/examples/navigation): A single agent navigating a large maze.
- [Detection Task](https://eyes.mit.edu/ACI/examples/detection): A single agent moving towards a target while avoiding an obstacle.
- [Tracking Task](https://eyes.mit.edu/ACI/examples/tracking): Similar to `detection`, but the target and obstacle move.
- [Optics](https://eyes.mit.edu/ACI/examples/optics): Outlines how to use our custom imaging model

### Train

To train a single agent in a detection-style task, you can run the following command. You will find the trained model and output files at `log/<today's date>/exp_detection`. This should take 10 to 20 minutes depending on your machine.

```bash
bash scripts/run.sh cambrian/main.py --train example=detection
```

After training, you can evaluate the agent using the following command. Output files will be saved in `log/<today's date>/exp_detection`.

```bash
python cambrian/main.py --eval example=detection trainer/model=loaded_model
```

## Documentation

For more detailed information on how to train, evaluate, and run experiments, see the [Documentation](https://eyes.mit.edu/ACI) website.

### Compiling the Documentation

First install the dev/doc dependencies.

```bash
pip install -e '.[doc,dev]'
```

Then to build the docs, run the following:

```bash
cd docs
make clean html
```

To view the build, go to your browser, and open the `index.html` file located inside `build/html/` (or run `open build/html/index.html`).

## Citation

```bibtex
@misc{aci,
    author = {Aaron Young and Kushagra Tiwary and Zaid Tasneem and Tzofi Klinghoffer and Bhavya Agrawalla and Sanjana Duttagupta and Akshat Dave and Brian Cheung},
    title = {{Artificial Cambrian Intelligence}},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/cambrian-org/ACI}},
}
```
