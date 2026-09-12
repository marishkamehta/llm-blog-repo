# From the blog to the repository

Read the [blog series](https://marishkamehta.github.io/blog/2026/controlled-experiments/).
Choose the corresponding topic below
to find its supporting code and results. Paths and commands assume one checkout of `llm-blog-repo`.

| Blog section | Files to use | What you can inspect |
| --- | --- | --- |
| [First behavioral experiment](https://marishkamehta.github.io/blog/2026/first-experiment/) | [Decoy example](../examples/README.md) | The same choice with and without a decoy; trial CSV |
| [Offline pipeline](https://marishkamehta.github.io/blog/2026/mock-pipeline/) | [Smoke test](../smokes/smoke_chat.py), [trajectory example](../examples/growing_trajectory.py) | Mock responses and conversation history |
| [Backend setup](https://marishkamehta.github.io/blog/2026/choosing-a-backend/) | [Example registry](../model-registry.yaml.example), [Docker example](../docker/docker-compose.yml) | Connection settings for your selected backend |
| [Position bias](https://marishkamehta.github.io/blog/2026/position-bias/) | [Workflow](../llm-behavior-demos/position_bias_demo/README.md), [material audit](../llm-behavior-demos/materials/position-bias/AUDIT.md), [results](../llm-behavior-demos/results/position-bias/README.md) | Swapped answer orders, parsing rules, repetition stability, summaries |
| [Belief bias](https://marishkamehta.github.io/blog/2026/belief-bias/) | [Workflow](../llm-behavior-demos/belief_bias_demo/README.md), [material audit](../llm-behavior-demos/materials/belief-bias/AUDIT.md), [results](../llm-behavior-demos/results/belief-bias/RESULTS.md) | Syllogism schedules, exclusions, accuracy and stability across temperatures |
| [Recording your own study](https://marishkamehta.github.io/blog/2026/experimental-vocabulary/) | [Reproducibility templates](../reproducibility/README.md) | Run manifests, artifact hashes, exclusions and deviations |

## A first visit

1. Install from the root using `python -m pip install -r requirements.txt`
   in an activated virtual environment.
2. Run `python smokes/smoke_chat.py --model mock`.
3. Read a demonstration's results and material audit before collecting data.
4. Follow that demonstration's README from inside `llm-behavior-demos/`.

The decoy example is a small introductory script. The position- and belief-bias
demonstrations provide the fuller schedule, collection, and analysis workflow.
Their source materials must be retrieved separately using the pinned sources
and checksums in the audits. Published summaries are available immediately;
recreating them requires the corresponding underlying model responses.
